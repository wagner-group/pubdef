from src.attack.base import AttackModule


class NoAttack(AttackModule):
    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )

    def forward(self, x, y):
        return x
