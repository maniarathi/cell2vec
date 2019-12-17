class Batch:
    def __init__(self, source, pad):
        self.source = source
        self.source_mask = (source != pad).unsqueeze(-2)
