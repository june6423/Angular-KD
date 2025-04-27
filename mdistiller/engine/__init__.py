from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, CRDAUGTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "crd_aug" : CRDAUGTrainer,
}
