这个是经验重现



# DERpp.py 代码详细解析

## 1. 整体架构

DERpp继承自`BaseLearner`，是一个基于经验回放的持续学习方法，通过教师-学生知识蒸馏来缓解灾难性遗忘。







核心其实通过两个办法

（1）一个就是暂存部分的旧数据，通过一个num_buffer_sample参数控制，num_buffer_sample前的数据得到的一个旧任务的CE损失（不遗忘），通过num_buffer_sample后的数据得到一个新任务的CE损失（学习新任务）

（2）再加一个教师模型和学生模型通过MSE均方差损失实现







## 2. 核心参数

```python
def get_DERpp_params(parser):
    parser.add_argument("--DERpp_alpha", type=float, default=0.5, 
                       help="The weight of MSE loss of buffer samples for DER++")
    parser.add_argument("--DERpp_beta", type=float, default=0.5, 
                       help="The weight of CE loss of buffer samples for DER++")
```

- `DERpp_alpha`: MSE蒸馏损失权重
- `DERpp_beta`: Buffer样本CE损失权重



关于这两个损失函数的设计原因

在DER++方法中，这两个损失权重发挥着不同但互补的作用：

 作用机制

ce_loss_old = self.ce_loss(student_logits[num_buffer_sample:], label_idx[num_buffer_sample:])

total_loss = ce_loss_new + self.params.DERpp_beta * ce_loss_old + self.params.DERpp_alpha * mse_loss_old

主要功能

- **保持分类能力**：确保模型对历史任务样本仍能正确分类
- **防止输出退化**：避免模型输出在旧任务上完全失效
- **维持决策边界**：保持模型在旧任务类别间的判别能力

具体作用

例如：模型在任务1学会了区分猫和狗

​		在学习任务2时，CE损失确保模型仍能：

将猫的图片分类为"猫"

\# - 将狗的图片分类为"狗"

\# 而不是输出错误的类别或无意义的概率分布

 MSE

```python
mse_loss_old = self.mse_loss(

  student_logits[num_buffer_sample:][:, :teacher_dims],

  teacher_logits[num_buffer_sample:]

)
```

**保持表征质量**：维持模型内部表征的稳定性**传递"暗知识"**：保留教师模型的细粒度知识**平滑过渡**：避免模型参数发生剧烈变化



## 3. 初始化约束

```python
def __init__(self, params, CL_dataset, accelerator):
    # 限制支持的配置
    assert params.classifier in ['None','Linear','CosineLinear']
    assert params.il_mode in ['CIL','TIL']
    assert params.classification_type in ['sentence-level']
    assert params.is_replay  # 必须启用经验回放
```

## 4. 关键组件构建

### 4.1 教师模型管理
```python
def begin_task(self, task_id):
    if task_id > 0:
        # 创建教师模型（当前模型的深拷贝）
        if self.wrap_teacher_model is not None:
            self.wrap_teacher_model.cpu()  # 释放GPU内存
            del self.wrap_teacher_model
        self.wrap_teacher_model = deepcopy(self.wrap_model)
```

### 4.2 Buffer更新
```python
def end_task(self, task_id):
    # 任务结束时更新buffer
    model = wrap_model.model
    self.buffer.update_buffer(task_id, self.train_loader_list[task_id], model, self.tokenizer)
```

## 5. 核心训练逻辑

### 5.1 数据合并策略

```python
def observe_batch(self, task_id, epoch_id, lm_input):
    if task_id > 0:
        # 从buffer采样历史数据
        buffer_lm_input = self.buffer.get_one_batch()
        # 合并新数据和历史数据
        lm_input = {k: torch.cat([lm_input[k], buffer_lm_input[k].to(lm_input[k].device)], dim=0) 
                   for k in lm_input.keys() if k in buffer_lm_input.keys()}
```

**关键设计**: 新数据在前，buffer数据在后，便于后续分离处理。

### 5.2 损失计算 - 生成式模型

```python
if self.params.classifier == 'None':  # Causal Language Model
    if task_id == 0:
        # 第一个任务：标准语言模型损失
        total_loss = model(**{'input_ids': lm_input['input_ids_with_ans'], 
                             'attention_mask': lm_input['attention_mask_with_ans'],
                             'labels': lm_input['labels_with_ans']}).loss
    else:
        # 计算教师模型预测（仅对buffer样本）
        num_buffer_sample = lm_input['input_ids_with_ans'].shape[0] // 2
        with torch.no_grad():
            teacher_logits = wrap_teacher_model.model(**{
                'input_ids': lm_input['input_ids_with_ans'][num_buffer_sample:], 
                'attention_mask': lm_input['attention_mask_with_ans'][num_buffer_sample:],
                'labels': lm_input['labels_with_ans'][num_buffer_sample:]
            }).logits
        
        # 学生模型预测（所有样本）
        student_logits = model(**{
            'input_ids': lm_input['input_ids_with_ans'], 
            'attention_mask': lm_input['attention_mask_with_ans'],
            'labels': lm_input['labels_with_ans']
        }).logits
        
        # 三个损失项
        # 1. 新样本CE损失 保证可以学到新任务的内容
        ce_loss_new = self.ce_loss(
            student_logits[:num_buffer_sample][:, :-1, :].contiguous().reshape(-1, student_logits.shape[-1]),
            lm_input['labels_with_ans'][:num_buffer_sample][:, 1:].contiguous().flatten()
        )
        
        # 2. Buffer样本CE损失 保证旧任务的内容不太多遗忘
        ce_loss_old = self.ce_loss(
            student_logits[num_buffer_sample:][:, :-1, :].contiguous().reshape(-1, student_logits.shape[-1]),
            lm_input['labels_with_ans'][num_buffer_sample:][:, 1:].contiguous().flatten()
        )
        
        # 3. Buffer样本MSE蒸馏损失 通过软标签进行匹配
        non_pad_mask = (lm_input['labels_with_ans'][num_buffer_sample:][:, 1:].contiguous() != -100)
        mse_loss_old = self.mse_loss(
            student_logits[num_buffer_sample:][:, :-1, :].contiguous()[non_pad_mask],
            teacher_logits[:, :-1, :].contiguous()[non_pad_mask]
        )
        
        # 总损失
        total_loss = ce_loss_new + self.params.DERpp_beta * ce_loss_old + self.params.DERpp_alpha * mse_loss_old
```

### 5.3 损失计算 - 判别式模型（CIL模式）

```python
elif self.params.classifier in ['Linear', 'CosineLinear']:
    # 提取特征
    extracted_feature = obtain_features(params=self.params, model=model, 
                                       lm_input=lm_input, tokenizer=self.tokenizer)
    
    if self.params.il_mode == 'CIL':
        # 连接所有任务的分类器输出
        student_logits = torch.concatenate([
            classifier(extracted_feature) for classifier in classifier_list[:task_id+1]
        ], dim=-1)
        
        if task_id == 0:
            total_loss = self.ce_loss(student_logits, label_idx)
        else:
            # 教师模型预测
            with torch.no_grad():
                teacher_features = obtain_features(params=self.params, 
                                                  model=wrap_teacher_model.model, 
                                                  lm_input=lm_input, 
                                                  tokenizer=self.tokenizer)
                teacher_logits = torch.concatenate([
                    classifier(teacher_features) for classifier in wrap_teacher_model.classifier_list[:task_id]
                ], dim=-1)
            
            # 三个损失项
            ce_loss_new = self.ce_loss(student_logits[:num_buffer_sample], label_idx[:num_buffer_sample])
            ce_loss_old = self.ce_loss(student_logits[num_buffer_sample:], label_idx[num_buffer_sample:])
            
            teacher_dims = teacher_logits.shape[-1]
            mse_loss_old = self.mse_loss(
                student_logits[num_buffer_sample:][:, :teacher_dims],
                teacher_logits[num_buffer_sample:]
            )
            
            total_loss = ce_loss_new + self.params.DERpp_beta * ce_loss_old + self.params.DERpp_alpha * mse_loss_old
```

### 5.4 损失计算 - 判别式模型（TIL模式）

```python
elif self.params.il_mode == 'TIL':
    total_loss = torch.tensor(0.).to(model.device)
    
    # 遍历所有任务
    for t_id in range(0, task_id+1):
        # 计算当前任务的类别范围
        class_idx_range_bg = self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][t_id]
        class_idx_range_ed = self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][t_id]
        
        # 找到属于当前任务的样本
        task_mask = torch.logical_and(
            lm_input['label_idx_cil'] >= class_idx_range_bg,
            lm_input['label_idx_cil'] < class_idx_range_ed
        )
        
        if task_mask.sum().item() == 0:
            continue
        
        # 使用任务特定的分类器
        student_logits = classifier_list[t_id](extracted_feature[task_mask])
        label_idx = lm_input['label_idx_til'][task_mask]
        
        if task_id == 0:
            total_loss += self.ce_loss(student_logits, label_idx)
        else:
            # 教师模型预测
            with torch.no_grad():
                teacher_features = obtain_features(params=self.params, 
                                                  model=wrap_teacher_model.model, 
                                                  lm_input=lm_input, 
                                                  tokenizer=self.tokenizer)
                teacher_logits = wrap_teacher_model.classifier_list[t_id](teacher_features[task_mask])
            
            ce_loss_old = self.ce_loss(student_logits, label_idx)
            mse_loss_old = self.mse_loss(student_logits, teacher_logits)
            
            total_loss += self.params.DERpp_beta * ce_loss_old + self.params.DERpp_alpha * mse_loss_old
```

## 6. 评估机制

### 6.1 生成式模型评估
```python
if classifier_list is None:
    acc = evaluate_sent_level_acc_with_generation(
        model=model,
        eval_data_loader=data_loader[eval_task_id],
        tokenizer=self.tokenizer,
        accelerator=self.accelerator,
        params=self.params,
        idx2label=self.CL_dataset.continual_config['idx2label']
    )
```

### 6.2 判别式模型评估
```python
elif self.params.classifier in ['Linear', 'CosineLinear']:
    acc = evaluate_sent_level_acc_with_classifier(
        model=model,
        classifier_list=classifier_list,
        cur_task_id=cur_task_id if self.params.il_mode=='CIL' else eval_task_id,
        eval_data_loader=data_loader[eval_task_id],
        tokenizer=self.tokenizer,
        accelerator=self.accelerator,
        params=self.params,
        idx2label=self.CL_dataset.continual_config['idx2label']
    )
```

## 7. 关键设计要点

### 7.1 内存管理
- 教师模型在任务切换时移至CPU并删除旧模型
- 使用`torch.cuda.empty_cache()`释放GPU内存

### 7.2 数据处理
- 新数据和buffer数据合并在同一个batch中
- 通过索引分离处理不同类型的样本

### 7.3 蒸馏策略
- **仅对buffer样本进行蒸馏**，避免对新样本的过度约束
- 使用MSE损失在logit级别进行知识蒸馏

### 7.4 损失平衡
- 三个损失项：新样本CE损失 + buffer样本CE损失 + buffer样本MSE蒸馏损失
- 通过α和β参数控制各损失项的权重

## 8. 优势与局限

### 优势
- ✅ 实现简单，效果稳定
- ✅ 支持多种模型类型和学习模式
- ✅ 良好的内存管理机制

### 局限
- ❌ 依赖经验回放，不适用于隐私敏感场景
- ❌ 需要额外存储教师模型，增加内存开销
- ❌ 每个batch需要计算教师模型预测，增加计算开销

DERpp作为持续学习的经典方法，在平衡简单性和有效性方面表现出色，是许多研究的重要基线。

Similar code found with 1 license type







