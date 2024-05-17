from beartype import beartype
import torch


@beartype
def huber_quant_reg_loss(
        td_errors: torch.Tensor, quantile: torch.Tensor, kappa: float = 1.) -> torch.Tensor:
    """Huber regression loss (introduced in 1964) following the definition
    in section 2.3 in the IQN paper (https://arxiv.org/abs/1806.06923).
    The loss involves a disjunction of 2 cases:
        case one: |td_errors| <= kappa
        case two: |td_errors| > kappa
    """
    aux = (0.5 * td_errors ** 2 *
           (torch.abs(td_errors) <= kappa).float() +
           kappa *
           (torch.abs(td_errors) - (0.5 * kappa)) *
           (torch.abs(td_errors) > kappa).float())
    return torch.abs(quantile - ((td_errors.le(0.)).float())) * aux / kappa
