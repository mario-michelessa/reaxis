from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


LOW_LEVEL_PREFIX = 'low_level_'


@dataclass(frozen=True)
class LowLevelFeatureSpec:
    field: str
    name: str
    query: str
    low_text: str
    high_text: str
    label_source: str
    ladder: tuple[str, str, str, str, str]

    @property
    def column(self) -> str:
        return f'{LOW_LEVEL_PREFIX}{self.field}'


LOW_LEVEL_FEATURES: tuple[LowLevelFeatureSpec, ...] = (
    LowLevelFeatureSpec(
        field='saturation',
        name='Saturation',
        query='color saturation',
        low_text='a low saturation image with muted gray colors',
        high_text='a high saturation image with vivid colors',
        label_source='mean_hsv_saturation',
        ladder=(
            'a very low saturation image',
            'a low saturation image',
            'a medium saturation image',
            'a high saturation image',
            'a very high saturation image',
        ),
    ),
    LowLevelFeatureSpec(
        field='brightness',
        name='Brightness',
        query='image brightness',
        low_text='a dark image',
        high_text='a bright image',
        label_source='mean_hsv_value',
        ladder=(
            'a very dark image',
            'a dark image',
            'a medium brightness image',
            'a bright image',
            'a very bright image',
        ),
    ),
    LowLevelFeatureSpec(
        field='contrast',
        name='Contrast',
        query='image contrast',
        low_text='a low contrast image',
        high_text='a high contrast image',
        label_source='std_grayscale_luminance',
        ladder=(
            'a very low contrast image',
            'a low contrast image',
            'a medium contrast image',
            'a high contrast image',
            'a very high contrast image',
        ),
    ),
    LowLevelFeatureSpec(
        field='sharpness',
        name='Sharpness',
        query='image sharpness',
        low_text='a blurry out of focus image',
        high_text='a sharp crisp image',
        label_source='variance_of_laplacian',
        ladder=(
            'a very blurry image',
            'a blurry image',
            'a moderately sharp image',
            'a sharp image',
            'a very sharp crisp image',
        ),
    ),
    LowLevelFeatureSpec(
        field='warmth',
        name='Warmth',
        query='color temperature warmth',
        low_text='an image with cool blue tones',
        high_text='an image with warm yellow red tones',
        label_source='mean_red_minus_blue',
        ladder=(
            'an image with very cool blue tones',
            'an image with cool tones',
            'an image with neutral color temperature',
            'an image with warm tones',
            'an image with very warm yellow red tones',
        ),
    ),
    LowLevelFeatureSpec(
        field='colorfulness',
        name='Colorfulness',
        query='image colorfulness',
        low_text='a muted color image',
        high_text='a colorful image',
        label_source='hasler_suesstrunk_colorfulness',
        ladder=(
            'a very muted color image',
            'a muted color image',
            'a moderately colorful image',
            'a colorful image',
            'a very colorful image',
        ),
    ),
    LowLevelFeatureSpec(
        field='edge_density',
        name='Edge density',
        query='edge density and visual detail',
        low_text='a smooth simple image with few edges',
        high_text='an image with many edges and details',
        label_source='canny_edge_pixel_fraction',
        ladder=(
            'a very smooth simple image',
            'a smooth image with few edges',
            'an image with a moderate number of edges',
            'an image with many edges',
            'an image with very many edges and details',
        ),
    ),
    LowLevelFeatureSpec(
        field='greenness',
        name='Greenness',
        query='green vegetation and greenery',
        low_text='an image with little greenery',
        high_text='an image with lots of greenery',
        label_source='mean_green_minus_red_blue_average',
        ladder=(
            'an image with almost no greenery',
            'an image with little greenery',
            'an image with some greenery',
            'an image with lots of greenery',
            'an image dominated by greenery',
        ),
    ),
)


def feature_columns() -> list[str]:
    return [spec.column for spec in LOW_LEVEL_FEATURES]


def _resolve_image_column(metadata: pd.DataFrame) -> str:
    stripped = {str(col).strip(): col for col in metadata.columns}
    if 'image' not in stripped:
        raise RuntimeError('metadata.csv must contain an image column')
    return str(stripped['image'])


def _load_rgb(path: Path, max_edge: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        if int(max_edge) > 0:
            rgb.thumbnail((int(max_edge), int(max_edge)), Image.Resampling.BICUBIC)
        return np.asarray(rgb, dtype=np.uint8)


def compute_low_level_feature_row(rgb_u8: np.ndarray) -> dict[str, float]:
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f'Expected RGB image array, got shape {rgb_u8.shape}')

    rgb01 = rgb_u8.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    gray_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gray01 = gray_u8.astype(np.float32) / 255.0

    r01 = rgb01[:, :, 0]
    g01 = rgb01[:, :, 1]
    b01 = rgb01[:, :, 2]
    r255 = rgb_u8[:, :, 0].astype(np.float32)
    g255 = rgb_u8[:, :, 1].astype(np.float32)
    b255 = rgb_u8[:, :, 2].astype(np.float32)

    rg = r255 - g255
    yb = (0.5 * (r255 + g255)) - b255
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    colorfulness += 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

    edges = cv2.Canny(gray_u8, threshold1=100, threshold2=200)

    return {
        'low_level_saturation': float(np.mean(hsv[:, :, 1].astype(np.float32) / 255.0)),
        'low_level_brightness': float(np.mean(hsv[:, :, 2].astype(np.float32) / 255.0)),
        'low_level_contrast': float(np.std(gray01)),
        'low_level_sharpness': float(cv2.Laplacian(gray_u8, cv2.CV_64F).var()),
        'low_level_warmth': float(np.mean(r01 - b01)),
        'low_level_colorfulness': float(colorfulness),
        'low_level_edge_density': float(np.mean(edges > 0)),
        'low_level_greenness': float(np.mean(g01 - (0.5 * (r01 + b01)))),
    }


def _needs_compute(metadata: pd.DataFrame, columns: Iterable[str], overwrite: bool) -> bool:
    if overwrite:
        return True
    missing = [col for col in columns if col not in metadata.columns]
    if missing:
        return True
    values = metadata[list(columns)].apply(pd.to_numeric, errors='coerce')
    return bool(values.isna().any(axis=None))


def compute_dataset_low_level_features(
    dataset_root: Path,
    *,
    overwrite: bool = False,
    max_edge: int = 512,
) -> Path:
    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f'Missing metadata.csv: {metadata_path}')

    metadata = pd.read_csv(metadata_path)
    image_column = _resolve_image_column(metadata)
    columns = feature_columns()
    if not _needs_compute(metadata, columns, overwrite):
        return metadata_path

    rows: list[dict[str, float]] = []
    image_names = metadata[image_column].astype(str).str.strip().tolist()
    for image_name in tqdm(image_names, desc=f'low-level features {dataset_root.name}'):
        image_path = dataset_root / image_name
        if not image_path.exists():
            raise FileNotFoundError(f'Metadata image not found: {image_path}')
        rows.append(compute_low_level_feature_row(_load_rgb(image_path, max_edge=max_edge)))

    feature_df = pd.DataFrame(rows)
    for column in columns:
        metadata[column] = feature_df[column].to_numpy(dtype=np.float64)
    metadata.to_csv(metadata_path, index=False)

    sidecar = dataset_root / 'low_level_features.csv'
    pd.concat([metadata[[image_column]].rename(columns={image_column: 'image'}), feature_df], axis=1).to_csv(
        sidecar,
        index=False,
    )
    return metadata_path


def compute_many_dataset_low_level_features(
    dataset_roots: Iterable[Path],
    *,
    overwrite: bool = False,
    max_edge: int = 512,
) -> list[Path]:
    out: list[Path] = []
    for root in dataset_roots:
        out.append(compute_dataset_low_level_features(Path(root), overwrite=overwrite, max_edge=max_edge))
    return out
