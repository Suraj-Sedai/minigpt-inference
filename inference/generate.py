import torch


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0
):
    model.reset_cache()

    # feed prompt tokens first
    for t in range(input_ids.size(1)):
        model.forward_step(input_ids[:, t:t+1])

    token = input_ids[:, -1:]

    outputs = [token]

    for _ in range(max_new_tokens):
        logits = model.forward_step(token)
        logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)

        outputs.append(token)

    return torch.cat(outputs, dim=1)
