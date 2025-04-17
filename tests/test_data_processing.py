import torch
import tempfile
import os
import pytest

from data_processing.data_processing import prepare_data

# Sample dummy text
dummy_text = "hello world. this is a test string." * 5

@pytest.fixture
def dummy_text_file():
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix=".txt") as f:
        f.write(dummy_text)
        f.flush()
        yield f.name
    os.remove(f.name)

def test_prepare_data_runs(dummy_text_file):
    data, tokenizer = prepare_data(dummy_text_file, batch_size=4, context_size=5)
    assert "train" in data
    assert "test" in data
    assert "x" in data["train"] and "y" in data["train"]
    assert "encoder" in tokenizer and "decoder" in tokenizer

def test_shapes_and_types(dummy_text_file):
    batch_size = 4
    context_size = 8
    data, _ = prepare_data(dummy_text_file, batch_size, context_size)

    for split in ["train", "test"]:
        x = data[split]["x"]
        y = data[split]["y"]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (batch_size, context_size)
        assert y.shape == (batch_size, context_size)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

def test_tokenizer_is_bijective(dummy_text_file):
    _, tokenizer = prepare_data(dummy_text_file, batch_size=2, context_size=4)
    enc = tokenizer["encoder"]
    dec = tokenizer["decoder"]

    # Ensure bijection
    for char, idx in enc.items():
        assert dec[idx] == char

def test_consistent_shapes_across_calls(dummy_text_file):
    b1, _ = prepare_data(dummy_text_file, batch_size=2, context_size=4)
    b2, _ = prepare_data(dummy_text_file, batch_size=2, context_size=4)

    assert b1["train"]["x"].shape == b2["train"]["x"].shape
    assert b1["train"]["y"].shape == b2["train"]["y"].shape
