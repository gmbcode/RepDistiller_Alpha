"""
train_nlp_crd.py (CORRECTED MEMORY BANK INDICES)
----------------
Contrastive Representation Distillation for NLP:
  Teacher: bert-base-uncased  (768-dim [CLS] embedding)
  Student: 2-layer Transformer OR BiLSTM
  Task:    SST-2 sentiment classification
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset

def parse_args():
    p = argparse.ArgumentParser(description="NLP Contrastive Representation Distillation")
    p.add_argument("--teacher",       type=str, default="bert-base-uncased")
    p.add_argument("--student_type",  type=str, default="transformer", choices=["transformer", "bilstm"])
    p.add_argument("--student_hidden", type=int, default=256)
    p.add_argument("--student_layers", type=int, default=2)
    p.add_argument("--student_heads",  type=int, default=4)
    p.add_argument("--task",          type=str, default="sst2")
    p.add_argument("--max_seq_len",   type=int, default=128)
    p.add_argument("--alpha",  type=float, default=0.0)
    p.add_argument("--beta",   type=float, default=0.8)
    p.add_argument("--nce_t",  type=float, default=0.07)
    p.add_argument("--nce_k",  type=int,   default=4096)
    p.add_argument("--nce_m",  type=float, default=0.5)
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--save_dir",      type=str,   default="./save/nlp_crd")
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--no_crd",  action="store_true")
    p.add_argument("--trial",   type=int, default=1)
    return p.parse_args()

TASK_CONFIG = {
    "sst2": {"dataset": ("nyu-mll/glue", "sst2"), "text_key": "sentence", "text_key2": None, "label_key": "label", "num_labels": 2},
}

class TransformerStudentClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, hidden=256, num_layers=2, num_heads=4, max_seq_len=128, dropout=0.1, teacher_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.proj = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, teacher_dim))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, num_labels))

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        tok_emb = self.embedding(input_ids)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = tok_emb + self.pos_embedding(pos_ids)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if attention_mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            pad_mask = (attention_mask == 0)
            src_key_padding_mask = torch.cat([cls_mask == 0, pad_mask], dim=1)
        else:
            src_key_padding_mask = None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_repr_raw = x[:, 0, :]
        return self.classifier(cls_repr_raw), self.proj(cls_repr_raw)

class BiLSTMStudentClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, hidden=256, num_layers=2, dropout=0.3, teacher_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        lstm_out_dim = hidden * 2
        self.proj = nn.Sequential(nn.Linear(lstm_out_dim, lstm_out_dim), nn.ReLU(), nn.Linear(lstm_out_dim, teacher_dim))
        self.classifier = nn.Sequential(nn.LayerNorm(lstm_out_dim), nn.Dropout(dropout), nn.Linear(lstm_out_dim, num_labels))

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            lengths = mask.sum(dim=1)
            pooled = (lstm_out * mask).sum(dim=1) / lengths
        else:
            pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled), self.proj(pooled)

class CRDLossNLP(nn.Module):
    def __init__(self, feat_dim, n_data, nce_k=4096, nce_t=0.07, nce_m=0.5):
        super().__init__()
        self.nce_k, self.nce_t, self.nce_m = nce_k, nce_t, nce_m
        self.register_buffer("memory_t", F.normalize(torch.randn(n_data, feat_dim), dim=1))
        self.n_data, self.feat_dim = n_data, feat_dim

    @torch.no_grad()
    def _update_memory(self, teacher_feat, indices):
        normed = F.normalize(teacher_feat.detach(), dim=1)
        self.memory_t[indices] = self.nce_m * self.memory_t[indices] + (1 - self.nce_m) * normed
        self.memory_t[indices] = F.normalize(self.memory_t[indices], dim=1)

    def forward(self, teacher_feat, student_feat, indices):
        B, D = teacher_feat.shape
        device = teacher_feat.device
        t_feat = F.normalize(teacher_feat, dim=1)
        s_feat = F.normalize(student_feat, dim=1)
        pos_sim = (s_feat * t_feat).sum(dim=1, keepdim=True) / self.nce_t
        K = min(self.nce_k, self.n_data - B)
        neg_indices = torch.randint(0, self.n_data, (B, K), device=device)
        neg_feat = self.memory_t[neg_indices]
        neg_sim = torch.bmm(neg_feat, s_feat.unsqueeze(-1)).squeeze(-1) / self.nce_t
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        self._update_memory(teacher_feat, indices)
        return loss

def load_and_tokenize(task, tokenizer, max_seq_len, batch_size, num_workers):
    cfg = TASK_CONFIG[task]
    raw = load_dataset(*cfg["dataset"])
    split, val_split = "train", "validation"
    
    def tokenize_fn(examples):
        encoded = tokenizer(examples[cfg["text_key"]], max_length=max_seq_len, truncation=True, padding="max_length")
        encoded["labels"] = examples[cfg["label_key"]]
        return encoded

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=raw[split].column_names)
    
    # FIXED: Hardcode permanent index for memory bank tracking
    def add_idx(example, idx):
        example["idx"] = idx
        return example
    tokenized = tokenized.map(add_idx, with_indices=True)
    
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels", "idx"])
    train_loader = DataLoader(tokenized[split], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(tokenized[val_split], batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(tokenized[split]), cfg["num_labels"]

@torch.no_grad()
def evaluate(student, val_loader, device):
    student.eval()
    correct = total = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = student(input_ids, attention_mask)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    student.train()
    return 100.0 * correct / total

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    train_loader, val_loader, n_data, num_labels = load_and_tokenize(args.task, tokenizer, args.max_seq_len, args.batch_size, args.num_workers)
    
    teacher = AutoModel.from_pretrained(args.teacher).to(device)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher_dim = teacher.config.hidden_size
    
    student = TransformerStudentClassifier(tokenizer.vocab_size, num_labels, args.student_hidden, args.student_layers, args.student_heads, args.max_seq_len + 1, teacher_dim=teacher_dim).to(device) if args.student_type == "transformer" else BiLSTMStudentClassifier(tokenizer.vocab_size, num_labels, args.student_hidden, args.student_layers, teacher_dim=teacher_dim).to(device)
    
    crd_loss_fn = CRDLossNLP(teacher_dim, n_data, args.nce_k, args.nce_t, args.nce_m).to(device)
    optimizer = AdamW(list(student.parameters()) + list(crd_loss_fn.parameters()), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.epochs * len(train_loader) * args.warmup_ratio), num_training_steps=args.epochs * len(train_loader))
    ce_loss_fn = nn.CrossEntropyLoss()
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = ce_loss = crd_loss_val = 0.0
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            indices = batch["idx"].to(device) # FIXED: Use hardcoded index, not sequential steps
            
            with torch.no_grad():
                teacher_repr = teacher(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            
            student_logits, student_repr = student(input_ids, attention_mask)
            loss_ce = ce_loss_fn(student_logits, labels)
            loss = loss_ce
            ce_loss += loss_ce.item()
            
            if args.beta > 0 and not args.no_crd:
                loss_crd = crd_loss_fn(teacher_repr, student_repr, indices)
                loss = loss + args.beta * loss_crd
                crd_loss_val += loss_crd.item()
                
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()
            
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch}/{args.epochs} step {step+1}/{len(train_loader)} loss={total_loss/(step+1):.4f} ce={ce_loss/(step+1):.4f} crd={crd_loss_val/(step+1):.4f}")
                
        val_acc = evaluate(student, val_loader, device)
        print(f"\n[Epoch {epoch}] val_acc = {val_acc:.2f}%")
        log_rows.append({"epoch": epoch, "val_acc": val_acc, "loss_ce": ce_loss / len(train_loader), "loss_crd": crd_loss_val / len(train_loader)})
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "val_acc": val_acc, "student_state": student.state_dict(), "args": vars(args)}, os.path.join(args.save_dir, f"{args.task}_{args.student_type}_best.pth"))

    import csv
    with open(os.path.join(args.save_dir, f"{args.task}_{args.student_type}_trial{args.trial}_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","val_acc","loss_ce","loss_crd"])
        writer.writeheader(); writer.writerows(log_rows)

if __name__ == "__main__":
    train(parse_args())