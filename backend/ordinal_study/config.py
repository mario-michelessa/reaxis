from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / 'data' / 'datasets'
OUTPUT_DIR = REPO_ROOT / 'backend' / 'experiments' / 'ordinal_modeling'

GLOBAL_SEED = 20260620
BUDGETS = (0, 1, 3, 5, 10, 20)
REPETITIONS = 5
MAX_PAIRWISE_METRIC_PAIRS = 2000
AXIS_BIN_COUNT = 10
QWK_CLASS_COUNT = 5

ACTIVE_DATASETS = ('UTKFace', 'AffectNet', 'KonIQ10k')
ACTIVE_METHODS = (
    'text_prior',
    'prompt_ladder',
    'ordinal_ridge',
    'rank_svm',
    'knn_ordinal',
    'kernel_ridge',
    'reaxis_random',
    'reaxis_active',
)
EXPANDED_DATASETS = (
    'SCUTFBP5500',
    'ISIC2024',
    'Messidor2',
    'VinDrMammo',
    'AADB',
    'LaMem',
    'OASIS',
    'HousePrices',
)

REAXIS_AXISBAYES_PARAMS = {
    'model_type': 'bayes_linear',
    'mode': 'gaussian',
    'feature_space': 'clip',
    'semantic_method': 'clip',
    'norm': True,
    'clip_weight': 1.0,
    'dino_weight': 0.0,
    'alpha': 0.34361647953904884,
    'dino_alpha': 0.0,
    'bias_alpha': 16.0,
    'sigma2': 0.04,
    'rank_eta': 1.3185334419106902,
    'rank_anchor_k': 3,
    'rank_anchor_delta': 0.3181991587400454,
    'rank_max_pairs': 224,
    'axis_bounds_text_source': 'template',
    'use_llm_prompt_ensemble': False,
}
REAXIS_RANK_PARAMS = {**REAXIS_AXISBAYES_PARAMS, 'mode': 'rank'}


@dataclass(frozen=True)
class AxisSpec:
    field: str
    name: str
    query: str
    low_text: str
    high_text: str
    min_value: float
    max_value: float
    label_source: str
    ladder: Tuple[str, str, str, str, str]
    category: str = 'high_level'


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    domain: str
    axes: Tuple[AxisSpec, ...]


def _axis(
    field: str,
    name: str,
    query: str,
    low_text: str,
    high_text: str,
    min_value: float,
    max_value: float,
    label_source: str,
    category: str,
) -> AxisSpec:
    return AxisSpec(
        field=field,
        name=name,
        query=query,
        low_text=low_text,
        high_text=high_text,
        min_value=min_value,
        max_value=max_value,
        label_source=label_source,
        ladder=(
            f'very {low_text}',
            low_text,
            f'medium {query}',
            high_text,
            f'very {high_text}',
        ),
        category=category,
    )


LOW_LEVEL_AXES: Tuple[AxisSpec, ...] = (
    _axis('low_level_saturation_01', 'Saturation', 'color saturation', 'low saturation muted gray image', 'high saturation vivid color image', 0.0, 1.0, 'mean_hsv_saturation_minmax', 'low_level'),
    _axis('low_level_brightness_01', 'Brightness', 'image brightness', 'dark image', 'bright image', 0.0, 1.0, 'mean_hsv_value_minmax', 'low_level'),
    _axis('low_level_contrast_01', 'Contrast', 'image contrast', 'low contrast image', 'high contrast image', 0.0, 1.0, 'std_grayscale_luminance_minmax', 'low_level'),
    _axis('low_level_sharpness_01', 'Sharpness', 'image sharpness', 'blurry out of focus image', 'sharp crisp image', 0.0, 1.0, 'variance_of_laplacian_minmax', 'low_level'),
    _axis('low_level_warmth_01', 'Warmth', 'color temperature warmth', 'cool blue toned image', 'warm yellow red toned image', 0.0, 1.0, 'mean_red_minus_blue_minmax', 'low_level'),
    _axis('low_level_colorfulness_01', 'Colorfulness', 'image colorfulness', 'muted color image', 'colorful image', 0.0, 1.0, 'hasler_suesstrunk_colorfulness_minmax', 'low_level'),
    _axis('low_level_edge_density_01', 'Edge density', 'edge density and visual detail', 'smooth simple image with few edges', 'detailed image with many edges', 0.0, 1.0, 'canny_edge_pixel_fraction_minmax', 'low_level'),
    _axis('low_level_greenness_01', 'Greenness', 'green vegetation and greenery', 'image with little greenery', 'image with lots of greenery', 0.0, 1.0, 'mean_green_minus_red_blue_average_minmax', 'low_level'),
)


def _with_low_level(*axes: AxisSpec) -> Tuple[AxisSpec, ...]:
    return tuple(axes) + LOW_LEVEL_AXES


def _with_low_level_subset(fields: Tuple[str, ...], *axes: AxisSpec) -> Tuple[AxisSpec, ...]:
    wanted = {f'low_level_{field}_01' for field in fields}
    return tuple(axes) + tuple(axis for axis in LOW_LEVEL_AXES if axis.field in wanted)


GRAYSCALE_LOW_LEVEL_FIELDS = ('brightness', 'contrast', 'sharpness', 'edge_density')


DATASET_SPECS: Dict[str, DatasetSpec] = {
    'UTKFace': DatasetSpec(
        name='UTKFace',
        root=DATASETS_ROOT / 'UTKFace',
        domain='Faces',
        axes=(
            AxisSpec(
                field='age',
                name='Age',
                query='visible age',
                low_text='a very young face',
                high_text='a very old face',
                min_value=0.0,
                max_value=116.0,
                label_source='filename_age',
                ladder=(
                    'a baby or toddler face',
                    'a young child face',
                    'a young adult face',
                    'a middle aged adult face',
                    'an elderly adult face',
                ),
            ),
        ),
    ),
    'AffectNet': DatasetSpec(
        name='AffectNet',
        root=DATASETS_ROOT / 'AffectNet',
        domain='Faces',
        axes=(
            AxisSpec(
                field='valence_proxy',
                name='Valence proxy',
                query='positive emotional valence',
                low_text='a face expressing negative emotion',
                high_text='a face expressing positive emotion',
                min_value=-1.0,
                max_value=1.0,
                label_source='emotion_category_circumplex_proxy',
                ladder=(
                    'a face expressing very negative emotion',
                    'a face expressing negative emotion',
                    'a face with neutral emotion',
                    'a face expressing positive emotion',
                    'a face expressing very positive emotion',
                ),
            ),
            AxisSpec(
                field='arousal_proxy',
                name='Arousal proxy',
                query='emotional arousal',
                low_text='a calm low arousal face',
                high_text='an intense high arousal face',
                min_value=0.0,
                max_value=1.0,
                label_source='emotion_category_circumplex_proxy',
                ladder=(
                    'a very calm face',
                    'a low intensity facial expression',
                    'a moderately expressive face',
                    'an intense facial expression',
                    'an extremely high arousal facial expression',
                ),
            ),
        ),
    ),
    'KonIQ10k': DatasetSpec(
        name='KonIQ10k',
        root=DATASETS_ROOT / 'KonIQ10k',
        domain='General images',
        axes=(
            AxisSpec(
                field='mos',
                name='Image quality',
                query='image quality',
                low_text='a very low quality photograph',
                high_text='a very high quality photograph',
                min_value=0.0,
                max_value=100.0,
                label_source='mean_opinion_score',
                ladder=(
                    'a very low quality photograph',
                    'a low quality photograph',
                    'an average quality photograph',
                    'a high quality photograph',
                    'a very high quality photograph',
                ),
            ),
        ),
    ),
}

DATASET_SPECS.update({
    'SCUTFBP5500': DatasetSpec(
        name='SCUTFBP5500',
        root=DATASETS_ROOT / 'SCUTFBP5500',
        domain='Faces',
        axes=_with_low_level(
            _axis('attractiveness_score', 'Attractiveness', 'facial attractiveness', 'less attractive face', 'more attractive face', 1.0, 5.0, 'SCUT_FBP5500_beauty_score', 'abstract'),
        ),
    ),
    'ISIC2024': DatasetSpec(
        name='ISIC2024',
        root=DATASETS_ROOT / 'ISIC2024',
        domain='Dermoscopic skin lesions',
        axes=_with_low_level(
            _axis('clin_size_long_diam_mm', 'Lesion diameter', 'lesion diameter', 'small lesion', 'large lesion', 1.0, 28.5, 'ISIC2024_clin_size_long_diam_mm', 'medium_level'),
            _axis('tbp_lv_areaMM2', 'Lesion area', 'lesion area', 'small lesion area', 'large lesion area', 0.0, 335.0, 'ISIC2024_tbp_lv_areaMM2', 'medium_level'),
            _axis('tbp_lv_eccentricity', 'Lesion eccentricity', 'elongated lesion shape', 'round lesion', 'elongated lesion', 0.0, 1.0, 'ISIC2024_tbp_lv_eccentricity', 'medium_level'),
            _axis('tbp_lv_symm_2axis', 'Lesion asymmetry', 'lesion asymmetry', 'symmetric lesion', 'asymmetric lesion', 0.0, 1.0, 'ISIC2024_tbp_lv_symm_2axis', 'medium_level'),
            _axis('tbp_lv_norm_border', 'Border irregularity', 'lesion border irregularity', 'regular smooth lesion border', 'irregular lesion border', 0.0, 10.0, 'ISIC2024_tbp_lv_norm_border', 'medium_level'),
            _axis('tbp_lv_norm_color', 'Color irregularity', 'lesion color irregularity', 'uniform lesion color', 'irregular lesion color', 0.0, 10.0, 'ISIC2024_tbp_lv_norm_color', 'medium_level'),
            _axis('tbp_lv_symm_2axis_angle', 'Lesion axis angle', 'lesion orientation angle', 'low angle lesion orientation', 'high angle lesion orientation', 0.0, 180.0, 'ISIC2024_tbp_lv_symm_2axis_angle', 'medium_level'),
        ),
    ),
    'Messidor2': DatasetSpec(
        name='Messidor2',
        root=DATASETS_ROOT / 'Messidor2',
        domain='Retinal fundus',
        axes=_with_low_level(
            _axis('diagnosis', 'Diabetic retinopathy severity', 'diabetic retinopathy severity', 'benign retina without diabetic retinopathy signs', 'severe diabetic retinopathy retina', 0.0, 4.0, 'MESSIDOR2_DR_grade', 'high_level'),
        ),
    ),
    'VinDrMammo': DatasetSpec(
        name='VinDrMammo',
        root=DATASETS_ROOT / 'VinDrMammo',
        domain='Mammography',
        axes=_with_low_level_subset(
            GRAYSCALE_LOW_LEVEL_FIELDS,
            _axis('breast_birads_grade', 'BI-RADS assessment', 'mammography BI-RADS severity', 'low suspicion mammogram', 'high suspicion mammogram', 1.0, 5.0, 'VinDr_breast_BIRADS', 'high_level'),
            _axis('breast_density_grade', 'Breast density', 'mammographic breast density', 'low density mammogram', 'high density mammogram', 0.0, 3.0, 'VinDr_breast_density', 'high_level'),
            _axis('finding_box_area_fraction', 'Finding size', 'mammography finding size', 'small or absent mammography finding', 'large mammography finding', 0.0, 1.0, 'VinDr_finding_box_area_fraction', 'medium_level'),
        ),
    ),
    'AADB': DatasetSpec(
        name='AADB',
        root=DATASETS_ROOT / 'AADB',
        domain='General photographs',
        axes=_with_low_level(
            _axis('aesthetic_score', 'Aesthetic score', 'photographic aesthetic quality', 'low aesthetic quality photograph', 'high aesthetic quality photograph', 0.0, 1.0, 'AADB_overall_score', 'abstract'),
            _axis('ColorHarmony', 'Color harmony', 'color harmony', 'disharmonious colors', 'harmonious colors', -1.0, 1.0, 'AADB_ColorHarmony', 'abstract'),
            _axis('DoF', 'Depth of field', 'depth of field quality', 'flat depth of field photograph', 'strong depth of field photograph', -1.0, 1.0, 'AADB_DoF', 'medium_level'),
            _axis('Light', 'Lighting quality', 'lighting quality', 'poorly lit photograph', 'well lit photograph', -1.0, 1.0, 'AADB_Light', 'medium_level'),
            _axis('RuleOfThirds', 'Rule of thirds', 'rule of thirds composition', 'weak rule of thirds composition', 'strong rule of thirds composition', -1.0, 1.0, 'AADB_RuleOfThirds', 'medium_level'),
            _axis('Symmetry', 'Symmetry', 'image symmetry', 'asymmetric composition', 'symmetric composition', 0.0, 1.0, 'AADB_Symmetry', 'medium_level'),
            _axis('VividColor', 'Vivid color', 'vivid color aesthetics', 'dull color photograph', 'vivid color photograph', -1.0, 1.0, 'AADB_VividColor', 'abstract'),
        ),
    ),
    'LaMem': DatasetSpec(
        name='LaMem',
        root=DATASETS_ROOT / 'LaMem',
        domain='General photographs',
        axes=_with_low_level(
            _axis('memorability_score', 'Memorability', 'image memorability', 'less memorable image', 'more memorable image', 0.0, 1.0, 'LaMem_memorability_score', 'abstract'),
        ),
    ),
    'OASIS': DatasetSpec(
        name='OASIS',
        root=DATASETS_ROOT / 'OASIS',
        domain='Emotion images',
        axes=_with_low_level(
            _axis('valence_mean', 'Valence', 'emotional valence', 'negative unpleasant image', 'positive pleasant image', 1.0, 7.0, 'OASIS_valence_mean', 'high_level'),
            _axis('arousal_mean', 'Arousal', 'emotional arousal', 'calm low arousal image', 'exciting high arousal image', 1.0, 7.0, 'OASIS_arousal_mean', 'high_level'),
            _axis('beauty_mean', 'Beauty', 'image beauty', 'less beautiful image', 'more beautiful image', 1.0, 7.0, 'OASIS_beauty_mean', 'abstract'),
        ),
    ),
    'HousePrices': DatasetSpec(
        name='HousePrices',
        root=DATASETS_ROOT / 'HousePrices',
        domain='House photographs',
        axes=_with_low_level(
            _axis('price_usd', 'House price', 'house price', 'low price house', 'high price house', 22000.0, 5858000.0, 'house_price_usd', 'abstract'),
            _axis('area_sqft', 'House area', 'house size', 'small house', 'large house', 700.0, 9600.0, 'house_area_square_feet', 'medium_level'),
            _axis('bedrooms', 'Bedrooms', 'number of bedrooms', 'few bedroom house', 'many bedroom house', 1.0, 10.0, 'house_bedrooms', 'medium_level'),
            _axis('bathrooms', 'Bathrooms', 'number of bathrooms', 'few bathroom house', 'many bathroom house', 1.0, 7.0, 'house_bathrooms', 'medium_level'),
        ),
    ),
})
