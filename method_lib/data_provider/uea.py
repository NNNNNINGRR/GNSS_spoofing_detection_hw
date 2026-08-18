import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class UEAloader(torch.utils.data.Dataset):
    """Lightweight UEA .ts loader for the Bi-FI reproduction.

    Parses the TSC long-format .ts files (multivariate or univariate) without
    requiring sktime/tslearn. Files expected: <Name>_TRAIN.ts / <Name>_TEST.ts
    inside root_path. Mirrors the interface used by TSL (feature_df,
    class_names, max_seq_len) so the existing classification experiment code
    works unchanged.
    """

    def __init__(self, root_path, flag="TRAIN", file_list=None, limit_size=None):
        super(UEAloader, self).__init__()
        self.root_path = root_path
        self.flag = flag
        name = os.path.basename(os.path.normpath(root_path))
        self.dataset_name = name
        split = "TRAIN" if "train" in str(flag).lower() else "TEST"
        ts_path = os.path.join(root_path, "{}_{}.ts".format(name, split))
        if not os.path.exists(ts_path):
            raise FileNotFoundError(ts_path)
        samples, raw_labels = _parse_uea_ts(ts_path)
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:
                limit_size = int(limit_size * len(samples))
            samples = samples[:limit_size]
            raw_labels = raw_labels[:limit_size]
        class_names = sorted(set(raw_labels))
        code_map = {c: i for i, c in enumerate(class_names)}
        labels = np.asarray([code_map[l] for l in raw_labels], dtype=np.int64)
        self.samples = [np.ascontiguousarray(x, dtype=np.float32) for x in samples]
        self.labels = labels
        self.max_seq_len = max(x.shape[0] for x in self.samples) if self.samples else 0
        self.class_names = np.asarray(class_names)
        self.feature_df = np.zeros((1, self.samples[0].shape[1]), dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def instance_norm(self, case):
        if "EthanolConcentration" in self.root_path:
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case = case / stdev
        return case

    def __getitem__(self, ind):
        x = torch.from_numpy(self.samples[ind])
        x = self.instance_norm(x)
        y = torch.tensor([self.labels[ind]], dtype=torch.long)
        return x, y


def _parse_uea_ts(path):
    """Parse a TSC .ts file into (samples, labels).

    Returns samples as a list of float arrays of shape (T_i, D) and labels as
    a 1-D numpy array of class values.
    """
    samples = []
    labels = []
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("@") or line.startswith("#"):
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            if not in_data:
                continue
            if "(" in line:
                channels = re.findall(r"\(([^)]*)\)", line)
                label_str = line[line.rfind(")") + 1:].strip().lstrip(":").strip()
            else:
                parts = line.split(":")
                label_str = parts[-1].strip()
                channels = parts[:-1] if len(parts) > 1 else [parts[0]]
            arrs = []
            for ch in channels:
                vals = []
                for v in ch.split(","):
                    v = v.strip()
                    if v in ("?", "NaN", "nan", ""):
                        vals.append(float("nan"))
                    else:
                        vals.append(float(v))
                arrs.append(np.asarray(vals, dtype=np.float64))
            tlen = max(len(a) for a in arrs)
            mat = np.full((tlen, len(arrs)), np.nan, dtype=np.float64)
            for j, a in enumerate(arrs):
                mat[:len(a), j] = a
            samples.append(mat)
            labels.append(label_str)
    return samples, labels
