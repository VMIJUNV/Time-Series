
class BaseDataclass:
    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)
    @classmethod
    def from_yaml(cls, path: str):
        import yaml
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))
