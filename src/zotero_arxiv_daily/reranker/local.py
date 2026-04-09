from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np
import pathlib
from huggingface_hub import snapshot_download
@register_reranker("local")
class LocalReranker(BaseReranker):
    @staticmethod
    def _patch_trust_remote_code(model_dir: pathlib.Path):
        """Fix duplicate trust_remote_code kwarg bug in jina model's custom_st.py.

        sentence-transformers 5.3.0 forwards trust_remote_code via kwargs to
        the custom module, whose __init__ captures it in **tokenizer_kwargs.
        The module then passes trust_remote_code=True explicitly *and* via
        **tokenizer_kwargs to AutoTokenizer.from_pretrained, causing a TypeError.
        We fix this by inserting a pop() call to remove the duplicate before it
        reaches from_pretrained.
        """
        custom_st_file = model_dir / "custom_st.py"
        if not custom_st_file.exists():
            return
        content = custom_st_file.read_text()
        if ("**tokenizer_kwargs" not in content
                or "trust_remote_code=True" not in content
                or "tokenizer_kwargs.pop('trust_remote_code'" in content):
            return
        lines = content.split('\n')
        patched_lines = []
        for line in lines:
            if 'AutoTokenizer.from_pretrained(' in line:
                indent = ' ' * (len(line) - len(line.lstrip()))
                patched_lines.append(f"{indent}tokenizer_kwargs.pop('trust_remote_code', None)")
            patched_lines.append(line)
        if custom_st_file.is_symlink():
            custom_st_file.unlink()
        custom_st_file.write_text('\n'.join(patched_lines))

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        model_dir = pathlib.Path(snapshot_download(self.config.reranker.local.model))
        self._patch_trust_remote_code(model_dir)
        encoder = SentenceTransformer(str(model_dir), trust_remote_code=True)
        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}
        s1_feature = encoder.encode(s1,**encode_kwargs)
        s2_feature = encoder.encode(s2,**encode_kwargs)
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
