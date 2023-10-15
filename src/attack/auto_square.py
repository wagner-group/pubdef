from autoattack import AutoAttack

from src.attack.base import AttackModule


class AutoAttackSP(AttackModule):
    def __init__(
        self,
        attack_config,
        core_model,
        loss_fn,
        norm,
        eps,
        verbose=False,
        num_classes=10,
        **kwargs
    ):
        super().__init__(
            attack_config,
            core_model,
            loss_fn,
            norm,
            eps,
            verbose=verbose,
            **kwargs
        )
        self.num_classes = num_classes

    def forward(self, x, y):
        mode = self._core_model.training
        self._core_model.eval()
        # TODO: Try to init adversary only once
        adversary = AutoAttack(
            self._core_model,
            norm=self._norm,
            eps=self._eps,
            version="standard-square+",
            verbose=self._verbose,
            # num_classes=self.num_classes
        )
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        self._core_model.train(mode)
        return x_adv
