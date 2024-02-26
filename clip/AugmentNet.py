import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO 实现SMLP的训练过程：TriangleCL、Loss(G(X))、Loss Consistency
# 1. 实现SMLP的训练过程：TriangleCL、Loss(G(X))、Loss Consistency
# 2. SMLP增强模型训练，并对比Loss Consis的方式的影响
# 3. 协同训练

class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.bn2 = nn.BatchNorm1d(input_size)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.bn2(self.linear2(x))
        x += residual  # Residual connection
        return F.relu(x)

class StackedMLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers=6):
        super(StackedMLP, self).__init__()

        self.mlp_layers = nn.ModuleList()
        prev_size = input_size
        hidden_size = input_size * 2

        # Create and stack MLP layers with residual connections
        for _ in range(n_layers):
            residual_block = ResidualBlock(prev_size, hidden_size)
            self.mlp_layers.append(residual_block)

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

        # Initialize parameters
        self.init_parameters()

    def init_parameters(self):
        for layer in self.mlp_layers:
            for param in layer.parameters():
                if len(param.shape) > 1:  # Initialize only linear layer parameters
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # Forward pass through each MLP layer
        for layer in self.mlp_layers:
            x = layer(x)

        # Output layer
        x = self.output_layer(x)
        return x

def get_samples(x_vector, y_vector, logit_scale, K, eta):
    bias_vector = y_vector - x_vector
    abs_bias_vector = torch.abs(bias_vector)
    W_r = (abs_bias_vector - torch.min(abs_bias_vector, axis=1, keepdims=True).values) / \
        (torch.max(abs_bias_vector, 1, keepdims=True).values -
         torch.min(abs_bias_vector, 1, keepdims=True).values)
    # initializing the set of samples
    R = []
    omega = torch.normal(0, W_r)
    sample = x_vector + torch.multiply(omega, bias_vector)
    loss = clip_loss(sample, y_vector, logit_scale)
    R.append(sample)
    for i in range(1, K):
        chain = [item.unsqueeze(dim=1) for item in R[:i]]
        average_omega = torch.mean(torch.cat(chain, dim=1), dim=1)
        omega = eta * torch.normal(0, W_r) + (1.0 - eta) * \
            torch.normal(average_omega, 1.0)
        sample = x_vector + torch.multiply(omega, bias_vector)
        loss += clip_loss(sample, y_vector, logit_scale)
        R.append(sample)
    return loss


def mgmc_sampling(src_embedding, trg_embedding, logit_scale, K=20, eta=0.6):
    # src_embedding: [batch_size, hidden_size]
    # trg_embedding: [batch_size, hidden_size]
    # default: K=20 and eta = 0.6
    loss_src = get_samples(src_embedding, trg_embedding, logit_scale, K, eta) / K
    loss_tgt = get_samples(trg_embedding, src_embedding, logit_scale, K, eta) / K
    return loss_src, loss_tgt

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(image_features, text_features, logit_scale) -> torch.Tensor:
    # cosine similarity as logits
    similarity = logit_scale * image_features @ text_features.t()
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0

def frozen(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False

class AugNet(nn.Module):

    def __init__(self, clip, n_layers, mode='task'):
        super().__init__()
        self.clip = clip
        embed_dim = self.clip.text_projection.shape[1]
        self.semantic_encoder = StackedMLP(embed_dim, embed_dim, n_layers)
        self.mode = mode
        if self.mode == 'aug':
            frozen(self.clip)
        elif self.mode == 'task':
            frozen(self.semantic_encoder)

    def encode_image(self, image):
        return self.clip.encode_image(image)
    
    def encode_text(self, text):
        return self.clip.encode_text(text)

    @property
    def logit_scale(self):
        return self.clip.logit_scale
    
    def infer(self, image, text):
        return self.clip(image, text)

    def forward(self, image, text, is_train=True):
        if not is_train:
            return self.infer(image, text)
        
        # mode in 'aug', 'task', 'union', 'infer'
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        contrast_acc= -1

        if self.mode == 'task':
            # 复现casnmt基于插值的最终增强向量使用
            image_features_aug = image_features * 0.5 + self.semantic_encoder(image_features) * 0.5
            text_features_aug = text_features * 0.5 + self.semantic_encoder(text_features) * 0.5

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            image_features_aug = image_features_aug / image_features_aug.norm(dim=1, keepdim=True)
            text_features_aug = text_features_aug / text_features_aug.norm(dim=1, keepdim=True)

            loss = clip_loss(image_features, text_features, logit_scale)
            loss_aug = (clip_loss(image_features_aug, text_features, logit_scale) + clip_loss(image_features, text_features_aug, logit_scale)) / 2
            return loss + loss_aug, contrast_acc
        elif self.mode == 'aug':
            # 复现casnmt ctl loss，在augnet训练阶段不加入额外损失
            image_features_aug = self.semantic_encoder(image_features)
            text_features_aug = self.semantic_encoder(text_features)

            image_features_aug = image_features_aug / image_features_aug.norm(dim=1, keepdim=True)
            text_features_aug = text_features_aug / text_features_aug.norm(dim=1, keepdim=True)

            loss_ctl, contrast_acc = get_ctl_loss(image_features_aug.unsqueeze(1), text_features_aug.unsqueeze(1), temperature=0.5)
            return loss_ctl, contrast_acc
        elif self.mode == 'union':
            # augnet与原网络协同训练，学习率较小。额外引入增强数据和源标签的损失，串联俩任务
            # normalized features
            image_features_aug = self.semantic_encoder(image_features)
            text_features_aug = self.semantic_encoder(text_features)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            image_features_aug = image_features_aug / image_features_aug.norm(dim=1, keepdim=True)
            text_features_aug = text_features_aug / text_features_aug.norm(dim=1, keepdim=True)
            
            loss = clip_loss(image_features, text_features, logit_scale)
            loss_aug = (clip_loss(image_features, text_features_aug, logit_scale) + clip_loss(image_features_aug, text_features, logit_scale)) / 2
            loss_ctl, contrast_acc = get_ctl_loss(image_features_aug.unsqueeze(1), text_features_aug.unsqueeze(1), temperature=0.5)
            return loss + (loss_aug + loss_ctl) * 0.25, contrast_acc

import torch
import torch.nn.functional as F

def get_ctl_loss(src_embedding, trg_embedding, dynamic_coefficient=1.0, temperature=1.0):
    # src_embedding: [batch_size, 1, hidden_size]
    # trg_embedding: [batch_size, 1, hidden_size]
    batch_size = src_embedding.size(0)

    def get_ctl_logits(query, keys):
        expand_query = query.repeat(1, batch_size, 1)
        expand_keys = keys.permute(1, 0, 2).repeat(batch_size, 1, 1)

        d_pos = torch.sqrt(torch.sum((query - keys)**2, dim=-1))  # [batch_size, 1]
        d_pos = d_pos.repeat(1, batch_size)  # [batch_size, batch_size]
        d_neg = torch.sqrt(torch.sum((expand_query - expand_keys)**2, dim=-1))  # [batch_size, batch_size]

        lambda_coefficient = (d_pos / d_neg)**dynamic_coefficient
        hardness_masks = torch.gt(d_neg, d_pos).float()

        hard_keys = (expand_query + lambda_coefficient.unsqueeze(2) * (expand_keys - expand_query)) * hardness_masks.unsqueeze(2) + \
                    expand_keys * (1.0 - hardness_masks.unsqueeze(2))

        logits = torch.bmm(query, hard_keys.transpose(1, 2)) / temperature  # [batch_size, 1, batch_size]
        return logits

    logits_src_trg = get_ctl_logits(src_embedding, trg_embedding)
    logits_trg_src = get_ctl_logits(trg_embedding, src_embedding) + \
                    torch.unsqueeze(torch.tril(torch.ones(batch_size, batch_size).to(src_embedding.device) * -1e9), dim=1)
    logits = torch.cat([logits_src_trg, logits_trg_src], dim=2)  # [batch_size, 1, 2*batch_size]

    labels = torch.unsqueeze(torch.arange(batch_size, dtype=torch.long), dim=1).to(src_embedding.device)

    loss = F.cross_entropy(logits.view(batch_size, 2*batch_size), labels.view(-1), reduction='mean')

    contrast_acc = torch.mean((torch.argmax(logits, dim=2) == labels.view(-1)).float()).item()

    return loss, contrast_acc


if __name__ == "__main__":

    # 定义输入、隐藏层和输出层的维度
    input_size = 512
    output_size = input_size

    # 创建多重堆叠MLP模型
    stacked_mlp = StackedMLP(input_size, output_size)

    # 打印模型结构
    print(stacked_mlp)

    import torch
    input = torch.rand(16, input_size)
    output = stacked_mlp(input)
    print(output.shape)
