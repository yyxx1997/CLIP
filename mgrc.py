import torch

def get_samples(x_vector, y_vector, K, eta):
    bias_vector = y_vector - x_vector
    abs_bias_vector = torch.abs(bias_vector)
    W_r = (abs_bias_vector - torch.min(abs_bias_vector, axis=1, keepdims=True).values) / \
        (torch.max(abs_bias_vector, 1, keepdims=True).values -
            torch.min(abs_bias_vector, 1, keepdims=True).values)
    # initializing the set of samples
    R = []
    omega = torch.normal(0, W_r)
    sample = x_vector + torch.multiply(omega, bias_vector)
    R.append(sample)
    for i in range(1, K):
        chain = [item.unsqueeze(dim=1) for item in R[:i]]
        average_omega = torch.mean(torch.cat(chain, dim=1), dim=1)
        omega = eta * torch.normal(0, W_r) + (1.0 - eta) * torch.normal(average_omega, 1.0)
        sample = x_vector + torch.multiply(omega, bias_vector)
        R.append(sample)
    return R

def mgmc_sampling(src_embedding, trg_embedding, K=20, eta=0.6):
    # src_embedding: [batch_size, hidden_size]
    # trg_embedding: [batch_size, hidden_size]
    # default: K=20 and eta = 0.6
    x_sample = get_samples(src_embedding, trg_embedding, K, eta)
    y_sample = get_samples(trg_embedding, src_embedding, K, eta)
    return x_sample, y_sample


import torch
import torch.nn.functional as F

def get_ctl_loss(src_embedding, trg_embedding, dynamic_coefficient, normalize=True, temperature=1.0):
    # src_embedding: [batch_size, 1, hidden_size]
    # trg_embedding: [batch_size, 1, hidden_size]
    batch_size = src_embedding.size(0)

    if normalize:
        src_embedding = F.normalize(src_embedding, dim=2)
        trg_embedding = F.normalize(trg_embedding, dim=2)

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
                    torch.unsqueeze(torch.tril(torch.ones(batch_size, batch_size) * -1e9), dim=1)
    logits = torch.cat([logits_src_trg, logits_trg_src], dim=2)  # [batch_size, 1, 2*batch_size]

    labels = torch.unsqueeze(torch.arange(batch_size, dtype=torch.long), dim=1)
    one_hot_labels = F.one_hot(labels, num_classes=2*batch_size).float()

    loss = F.cross_entropy(logits.view(batch_size, 2*batch_size), labels.view(-1), reduction='mean')

    contrast_acc = torch.mean((torch.argmax(logits, dim=2) == labels.view(-1)).float())

    return loss, contrast_acc


if __name__ == "__main__":
    src = torch.rand(2, 3)
    tgt = torch.rand(2, 3)
    results = mgmc_sampling(src, tgt, K=20, eta=0.6)
    ...
    # mgrc sampling augmentation
    # extra_image_features, extra_text_features = mgmc_sampling(image_features, text_features, K=20, eta=0.6)  
    # extra_image_features = torch.cat(extra_image_features, dim=0)
    # extra_text_features = torch.cat(extra_text_features, dim=0)
    # image_features = torch.cat((image_features, extra_image_features), dim=0)
    # text_features = torch.cat((text_features, extra_text_features), dim=0)