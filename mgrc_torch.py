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