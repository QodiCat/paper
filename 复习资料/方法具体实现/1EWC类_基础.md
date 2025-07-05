# EWC (Elastic Weight Consolidation) 方法详解

## 1. 核心思想

EWC是一种基于**贝叶斯理论**的持续学习方法，通过计算参数的**Fisher信息矩阵**来识别对旧任务重要的参数，并在学习新任务时对这些重要参数施加正则化约束。

## 2. 理论基础

### 贝叶斯公式
```
P(θ|D) = P(D|θ)P(θ) / P(D)
```

### Fisher信息矩阵
Fisher信息矩阵衡量参数对损失函数的**敏感性**：
```python
F_ii = E[(∂log P(D|θ)/∂θ_i)²]
```

## 3. 核心参数

```python
def get_EWC_params(parser):
    parser.add_argument("--EWC_lambda", type=float, default=5000, 
                       help="The weight for the regularization term")
```

- `EWC_lambda`: 正则化项权重，控制对旧任务参数的保护强度

## 4. 关键组件

### 4.1 模型初始化
```python
def build_backbone(self):
    self.model, self.tokenizer = get_backbone(self.params)
    self.old_model = None  # 存储旧任务的模型
    self.fisher = None     # 存储Fisher信息矩阵
```

### 4.2 支持的配置
```python
def __init__(self, params, CL_dataset, accelerator):
    # EWC只支持特定配置
    assert params.classifier in ['None']           # 仅支持生成式模型
    assert params.il_mode in ['IIL']              # 仅支持IIL模式
    assert params.classification_type in ['sentence-level']
```

## 5. Fisher矩阵计算

### 5.1 核心算法
```python
def fisher_matrix_diag(self, dataloader):
    '''计算Fisher信息矩阵的对角线元素'''
    
    # 初始化Fisher矩阵
    fisher = {}
    for n, p in self.model.named_parameters():
        fisher[n] = 0 * p.data  # 与参数同形状的零矩阵
    
    total_cnt = 0
    for lm_input in dataloader:
        batch_size = lm_input['input_ids_with_ans'].shape[0]
        total_cnt += batch_size
        
        # 前向和反向传播
        self.model.zero_grad()
        loss = self.model(**{
            'input_ids': lm_input['input_ids_with_ans'], 
            'attention_mask': lm_input['attention_mask_with_ans'],
            'labels': lm_input['labels_with_ans']
        }).loss
        loss.backward()
        
        # 累积梯度的平方
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                fisher[n] += batch_size * p.grad.data.pow(2)
    
    # 计算平均值
    for n, _ in self.model.named_parameters():
        fisher[n] = fisher[n] / total_cnt
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
    
    return fisher
```

### 5.2 计算原理
- **梯度平方**: `p.grad.data.pow(2)` 计算每个参数梯度的平方
- **批次加权**: `batch_size * p.grad.data.pow(2)` 按批次大小加权
- **平均化**: 除以总样本数得到期望值
- **对角近似**: 只计算对角线元素，忽略参数间的协方差

## 6. 任务级别操作

### 6.1 任务结束处理
```python
def end_task(self, task_id):
    # 保存当前模型作为"旧模型"
    if self.old_model is not None:
        self.old_model.cpu()
        del self.old_model
    self.old_model = deepcopy(self.model)
    
    # 处理Fisher矩阵的累积
    if task_id > 0: 
        fisher_old = {}
        for n, _ in self.model.named_parameters():
            fisher_old[n] = self.fisher[n].clone()
    
    # 计算当前任务的Fisher矩阵
    self.fisher = self.fisher_matrix_diag(self.train_loader_list[task_id])
    
    # 合并Fisher矩阵（加权平均）
    if task_id > 0:
        for n, _ in self.model.named_parameters():
            self.fisher[n] = (self.fisher[n] + fisher_old[n] * task_id) / (task_id + 1)
```

### 6.2 Fisher矩阵累积策略
```python
# 加权平均公式
fisher_new = (fisher_current + fisher_old * task_id) / (task_id + 1)
```

这种策略确保：
- 新任务的Fisher信息得到考虑
- 历史任务的Fisher信息不会被遗忘
- 随着任务数量增加，单个任务的影响逐渐降低

## 7. 损失函数设计

### 7.1 第一个任务
```python
if task_id == 0:
    total_loss = model(**{
        'input_ids': lm_input['input_ids_with_ans'], 
        'attention_mask': lm_input['attention_mask_with_ans'],
        'labels': lm_input['labels_with_ans']
    }).loss
```

### 7.2 后续任务
```python
else:
    # 1. 标准的因果语言建模损失
    causal_lm_loss = model(**{
        'input_ids': lm_input['input_ids_with_ans'], 
        'attention_mask': lm_input['attention_mask_with_ans'],
        'labels': lm_input['labels_with_ans']
    }).loss
    
    # 2. EWC正则化损失
    regularization_loss = 0
    for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                            self.old_model.named_parameters()):
        regularization_loss += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
    
    # 3. 总损失
    total_loss = causal_lm_loss + self.params.EWC_lambda * regularization_loss
```

### 7.3 正则化项解释


Similar code found with 1 license type