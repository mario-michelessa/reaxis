#!/usr/bin/env python3
from __future__ import annotations

# Backend server runtime constants.
BACKEND_HOST = '0.0.0.0'
BACKEND_PORT = 5001

# Initial gallery projection shown by the UI when no explicit /gallery.json
# `method` query is provided.
INITIAL_GALLERY_PROJECTION_METHOD = 'pca'  # 'pca' | 'umap' | 'tsne'


# LLM provider configuration.
# - huggingface_local: local model loaded from HF_LOCAL_MODEL_PATH
# - gemini_api: Google Gemini API using a locally stored API key file
LLM_PROVIDER = 'gemini_api'

# Gemini API configuration.
# Keep the API key in an ignored local file, not in tracked source.
# GEMINI_MODEL_NAME = 'gemma-3-27b'
GEMINI_MODEL_NAME = 'gemini-2.5-flash-lite'
# GEMINI_MODEL_NAME = 'gemini-2.5-flash'
# GEMINI_MODEL_NAME = 'gemini-3-flash'
GEMINI_API_KEY_PATH = 'data/secrets/gemini_api_key.txt'
GEMINI_API_TIMEOUT_SEC = 45

# Local Hugging Face LLM configuration.
# You can hardcode a path here, e.g.:
# HF_LOCAL_MODEL_PATH = "/models/Qwen2.5-3B-Instruct"
HF_LOCAL_MODEL_PATH = '/mnt/raid/mario/models/llms-theory/Qwen/Qwen2.5-3B-Instruct'
HF_LOCAL_FILES_ONLY = True
HF_TRUST_REMOTE_CODE = False
HF_MAX_NEW_TOKENS = 256
HF_TEMPERATURE = 0.2
HF_TOP_P = 0.95
HF_USE_4BIT = True
HF_4BIT_QUANT_TYPE = 'nf4'
HF_4BIT_COMPUTE_DTYPE = 'float16'
HF_4BIT_USE_DOUBLE_QUANT = True
HF_4BIT_DEVICE_MAP = 'auto'

# Extraction/value defaults.
DEFAULT_MAX_ATTRIBUTES = 8
DEFAULT_VALUE_COUNT = 5
MAX_VALUE_COUNT = 9

# Short dataset descriptions used as LLM context for axis suggestion and
# prompt-ensemble generation.
DATASET_LLM_CONTEXT: dict[str, str] = {
    'CUB': 'Photos of birds spanning many species, poses, plumage patterns, and natural backgrounds.',
    'EmoSet': 'Images labeled by emotion, including people, objects, scenes, and symbolic visuals designed to evoke feelings.',
    'HAM10000': 'Dermoscopy close-ups of skin lesions with diagnosis-relevant color, border, and texture variation.',
    'HubbleStars': 'Hubble astronomy images focused on stars, stellar clusters, nebulae, and bright celestial structures.',
    'ISIC2017': 'Dermoscopy images of skin lesions used for melanoma-related visual assessment.',
    'ISIC2020': 'Dermoscopy images of skin lesions with clinically relevant variation in pigmentation, borders, and structure.',
    'ImageNet': 'Natural photographs of diverse everyday objects, animals, and scenes across many categories.',
    'ImageNet_R': 'Artistic, rendered, or stylized depictions of ImageNet object classes rather than plain natural photos.',
    'ImageNet_n029583': 'A single ImageNet object class collection, so useful axes should focus on within-class visual variation.',
    'Imagenette1500': 'Natural photos from ten broad object classes in the Imagenette subset.',
    'MapillaryVistas': 'Street-scene photos with roads, cars, signs, buildings, sidewalks, and urban outdoor layouts.',
    'VIS30K': 'Visualization images such as charts, diagrams, plots, maps, and other designed graphics.',
    'VIS30KGUI': 'Information visualization and interface images, including charts, dashboards, and GUI-like visual designs.',
    'WikiArt1500': 'Artwork images spanning many painting styles, subjects, palettes, and compositions.',
    'ancient_tamil_inscriptions': 'Photos of ancient Tamil stone inscriptions with variation in carving, erosion, and surface appearance.',
    'archaeomind_images': 'Archaeological imagery containing artifacts and non-artifacts with varied materials, shapes, and excavation context.',
    'brain_mri_images': 'Brain MRI scans with medical structure, intensity, and possible tumor-related variation.',
    'broden1_224': 'Everyday images with richly varied scenes, objects, parts, materials, and textures.',
    'celeba_dataset': 'Face photos with variation in identity, hairstyle, expression, accessories, and facial attributes.',
    'chartqa_images': 'Chart and plot images used for chart question answering, with bars, lines, legends, and labels.',
    'chest_xray_pneumonia': 'Chest X-ray images with medically relevant lung opacity and anatomy variation.',
    'egyptian_stone_statues': 'Photos of Egyptian stone statues with variation in pose, damage, carving style, and material weathering.',
    'inat2021birds': 'Bird photographs in natural environments with strong variation in species, pose, color, and habitat.',
    'metal_albums_artwork': 'Album-cover artworks with strong variation in style, typography, mood, and graphic composition.',
    'paintings_wikiart': 'Paintings across artists, genres, and styles with wide variation in brushwork, subject, and color.',
    'qajar_carpets': 'Images of Qajar carpets with variation in motifs, symmetry, palette, and ornamental density.',
    'sinhala_brahmi_inscriptions': 'Photos of Sinhala Brahmi inscriptions carved into stone with variation in script, wear, and contrast.',
    'stars_hubble': 'Hubble astronomy images focused on stars, stellar clusters, nebulae, and bright celestial structures.',
}

# Zero-shot regressor defaults.
ZERO_SHOT_SOFTMAX_TEMPERATURE = 12.0
ZERO_SHOT_HISTOGRAM_BINS = 20

# Axis Bayes defaults.
# Scorer backend:
# - 'bayes_linear': legacy axis engine path (gaussian / rank / graph modes stay unchanged)
# - 'piecewise_linear': mixture-of-linear ranker trained mainly from ordering constraints
# - 'residual': global semantic text prior plus a small smooth GP residual
AXIS_MODEL_TYPE = 'bayes_linear'  # 'bayes_linear' | 'piecewise_linear' | 'residual'

# Update mode:
# - 'gaussian': numeric target updates in projection units for a global linear axis
# - 'rank': pairwise ranking updates for a global linear axis
# - 'graph': Bayesian Gaussian random field over per-image latent scores on a kNN graph
AXIS_BAYES_MODE = 'rank'  # 'gaussian' | 'rank' | 'graph'

# Semantic VLM block used by Axis Bayes.
# - 'siglip2': preferred semantic cache / text encoder
# - 'clip': compatibility fallback
AXIS_BAYES_SEMANTIC_METHOD = 'siglip2'  # 'siglip2' | 'clip'

# Whether Axis Bayes should use the normalized semantic caches directly.
# When False, raw semantic vectors are preserved in state and cosine normalization
# happens only at similarity-evaluation time inside the Bayes scorer.
AXIS_BAYES_NORM = False

# Feature space used by Axis Bayes.
# Legacy names are kept for compatibility:
# - 'clip': semantic VLM block only
# - 'clip_dino': semantic VLM block concatenated with DINO
AXIS_BAYES_FEATURE_SPACE = 'clip'  # 'clip' | 'clip_dino'

# Relative contribution of semantic-VLM and DINO blocks in fused features.
# Effective scaling uses normalized weights:
#   x_fused = [sqrt(w_clip)*x_semantic, sqrt(w_dino)*x_dino]
# These defaults follow the current best rank-mode sweep winner (Trial 69).
AXIS_BAYES_CLIP_WEIGHT = 0.8041519207887364
AXIS_BAYES_DINO_WEIGHT = 0.19584807921126357

# Piecewise-linear ranker controls.
# The new scorer still uses the existing CLIP/DINO embeddings, but can combine
# a small number of local linear experts instead of a single global direction.
AXIS_PIECEWISE_NUM_EXPERTS = 3
AXIS_PIECEWISE_USE_GATING = True
AXIS_PIECEWISE_AGGREGATOR = 'max'  # 'max' | 'mean' | 'softmax'

# Additional per-block scaling for the piecewise scorer.
# These are applied after recovering the normalized CLIP / DINO features from the
# legacy fused cache, so 1.0 keeps the current feature magnitudes unchanged.
AXIS_PIECEWISE_CLIP_SCALE = 1.3508877482680062
AXIS_PIECEWISE_DINO_SCALE = 0.22659722214371478

# Minimum scalar-label gap required before two moved examples become an explicit
# ranking pair. Smaller values create more constraints from close labels.
AXIS_PIECEWISE_PAIRWISE_FROM_SCALAR_MARGIN = 0.34936776596459196

# Regularization toward the text prior direction for expert 1.
AXIS_PIECEWISE_PRIOR_STRENGTH = 0.11917005611546902

# Penalizes expert collapse by discouraging similar expert directions.
AXIS_PIECEWISE_EXPERT_DIVERSITY_STRENGTH = 0.5197325862847438

# Standard L2 penalty on all expert weights.
AXIS_PIECEWISE_L2_REG = 0.04912946914323385

# Optimizer settings for each refinement after a user move.
AXIS_PIECEWISE_LEARNING_RATE = 0.07989580454753624
AXIS_PIECEWISE_MAX_REFINE_STEPS = 48
# Prior precision for CLIP block (all CLIP dimensions).
# Larger -> stronger pull toward CLIP text prior; smaller -> faster adaptation.
# Current default follows the best rank-mode sweep winner (Trial 69).
AXIS_BAYES_ALPHA = 1.64361647953904884

# Prior precision for DINO block (all DINO dimensions) when using 'clip_dino'.
# Larger than AXIS_BAYES_ALPHA keeps DINO conservative unless moves support it.
AXIS_BAYES_DINO_ALPHA = 0.41751659396568375

# Intercept (bias) prior precision.
# Smaller -> easier global shift of scores; larger -> stays closer to prior centering.
AXIS_BAYES_BIAS_ALPHA = 0.02

# Observation noise variance for gaussian mode.
# Larger values damp updates; smaller values trust move targets more.
AXIS_BAYES_SIGMA2 = 0.0006123800590318

# Graph-mode prior over per-image latent scores:
#   f ~ N(z0, (lambda_smooth * L + lambda_prior * I)^-1)
# where L is the symmetric kNN graph Laplacian built from the chosen feature space.
# Increase KNN_K to spread information across a broader local neighborhood.
AXIS_BAYES_GRAPH_KNN_K = 4

# Larger -> stronger manifold smoothing across nearby images on the graph.
AXIS_BAYES_GRAPH_LAMBDA_SMOOTH = 3.1392207160296035

# Larger -> keep posterior scores closer to the text/VLM prior z0.
AXIS_BAYES_GRAPH_LAMBDA_PRIOR = 0.050446153266139

# Small positive ridge for numerical stability when inverting graph precision matrices.
AXIS_BAYES_GRAPH_JITTER = 1e-6

# User-move trust (gaussian, rank, and graph modes).
# This sets how strongly each drag/move should pull the axis.
# Effective trust per move:
#   trust = AXIS_BAYES_MOVE_TRUST + AXIS_BAYES_MOVE_MAG_GAIN * |target - current|
# where target/current are percentile values in [0, 1].
AXIS_BAYES_MOVE_TRUST = 14.0
AXIS_BAYES_MOVE_MAG_GAIN = 36.0

# Rank-mode temperature eta in pairwise logistic likelihood.
# Smaller -> sharper comparisons, larger -> softer comparisons.
# Current default follows the best rank-mode sweep winner (Trial 69).
AXIS_BAYES_RANK_ETA = 1.3185334419106902

# Rank-mode anchor sampling around target percentile after a move.
# K controls anchors per side; DELTA controls percentile window radius.
AXIS_BAYES_RANK_ANCHOR_K = 3
AXIS_BAYES_RANK_ANCHOR_DELTA = 0.3181991587400454

# Maximum stored pairwise constraints per axis in rank mode.
# Higher can improve stability but increases compute.
AXIS_BAYES_RANK_MAX_PAIRS = 224

# Residual-mode controls: global CLIP text prior plus a small smooth GP residual.
# All values are intentionally conservative so the residual only bends the prior
# near labeled examples and falls back cleanly elsewhere.
AXIS_RESIDUAL_ALPHA = 1.0
AXIS_RESIDUAL_BETA = 0.0
AXIS_RESIDUAL_LAMBDA = 0.12
AXIS_RESIDUAL_SIGMA_Y = 0.06
AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER = 1.0
AXIS_RESIDUAL_JITTER = 1e-6

# Optional hard cap on number of image moves per axis (0 = unlimited).
AXIS_BAYES_MAX_MOVES = 0

# Hotspot selection controls (used for uncertainty-guided suggestions).
# boundary: target percentile center; tau: distance decay; k: number of hotspots.
AXIS_BAYES_HOTSPOT_BOUNDARY = 50.0
AXIS_BAYES_HOTSPOT_TAU = 18.0
AXIS_BAYES_HOTSPOT_K = 10

# Number of decile exemplars chosen per bin.
AXIS_BAYES_EXEMPLAR_K = 4

# Global prompt templates.
ATTRIBUTE_EXTRACTION_SYSTEM_PROMPT = """
You are an expert in visual analytics and vision-language semantics.

Task:
Given a user request about images (search, browse, compare, cluster, sort, filter, explain),
extract 3 to 10 concise visual attributes that:
- can plausibly vary across images in the described collection
- are observable in the pixels (or strongly correlated with pixels)
- are measurable using CLIP-like embeddings or simple vision estimators
- are useful as interactive controls (filtering, sorting, faceting, axes)

Output rules (strict):
- Return ONLY valid JSON (no prose, no markdown).
- Format exactly:
  {"attributes": [{"name": "...", "type": "categorical|ordinal|continuous"}]}
- Each attribute:
  - name is 1 to 4 words, lowercase, no punctuation
  - no duplicates or near-duplicates (merge synonyms)
  - avoid vague words like "aesthetic", "quality", "interesting", "nice", "good"
  - avoid implementation terms like "clip score", "embedding", "model confidence"

Type guidelines:
- categorical: unordered classes (e.g., lesion type, object category, setting)
- ordinal: ordered discrete states (e.g., low medium high; mild moderate severe)
- continuous: smooth scale (e.g., brightness, size, saturation, density)

Selection heuristics:
- Use common knowledge attributes depending on the domain.
- If the request implies a comparison axis, include that axis as an attribute.
- If the request mentions absence or exclusion (e.g., "no text", "without people"),
  convert it into a positive measurable attribute (e.g., "text presence", "people count").
- If the request mentions relationships (e.g., "near", "behind", "wearing"),
  include at most one relational attribute when it matters (e.g., "object proximity").
- Keep names dataset-agnostic unless the request is clearly domain-specific
  (e.g., dermoscopy terms are allowed for skin lesions).

Example 1
USER REQUEST:
"Let me explore dermoscopy photos and separate benign vs malignant, focusing on ABCD cues."
ASSISTANT OUTPUT:
{"attributes":[
  {"name":"diagnosis label","type":"categorical"},
  {"name":"asymmetry level","type":"ordinal"},
  {"name":"border irregularity","type":"ordinal"},
  {"name":"color variegation","type":"ordinal"},
  {"name":"lesion diameter","type":"continuous"},
  {"name":"blue white veil","type":"categorical"}
]}

Example 2
USER REQUEST:
"Find street photos with fewer distractions, strong negative space, and warm light at night."
ASSISTANT OUTPUT:
{"attributes":[
  {"name":"scene type","type":"categorical"},
  {"name":"clutter level","type":"ordinal"},
  {"name":"negative space","type":"continuous"},
  {"name":"color temperature","type":"continuous"},
  {"name":"time of day","type":"categorical"},
  {"name":"text presence","type":"categorical"}
]}

Example 3
USER REQUEST:
"Browse product shots and filter for front-facing, centered object, plain background, no hands."
ASSISTANT OUTPUT:
{"attributes":[
  {"name":"product category","type":"categorical"},
  {"name":"viewpoint angle","type":"continuous"},
  {"name":"object centering","type":"continuous"},
  {"name":"background complexity","type":"ordinal"},
  {"name":"hand presence","type":"categorical"}
]}
""".strip()
ATTRIBUTE_EXTRACTION_USER_PROMPT_TEMPLATE = """
USER REQUEST:
{prompt}

""".strip()

ATTRIBUTE_SUPPORT_SYSTEM_PROMPT = """
You explain why visual attributes were extracted from a user request.

Task:
For each provided attribute, identify the exact words or short phrases in the
user request that support that attribute.

Output rules (strict):
- Return ONLY valid JSON.
- Format exactly:
  {"supports": [{"name": "...", "spans": [{"text": "...", "score": 0.0}]}]}
- Each `name` must match one provided attribute name.
- Each `text` must be copied verbatim from the user request.
- Each attribute should have 1 to 3 support spans when possible.
- `score` must be between 0.0 and 1.0 and reflects how strongly the span
  supports the attribute.
- Prefer short, specific spans over long sentences.
- Do not invent text that is not present in the request.
- No prose, no markdown, no extra keys.

Example 1
USER REQUEST:
"Find street photos with fewer distractions, strong negative space, and warm light at night."
ATTRIBUTES:
[{"name":"clutter level","type":"ordinal"},{"name":"negative space","type":"continuous"},{"name":"color temperature","type":"continuous"},{"name":"time of day","type":"categorical"}]
OUTPUT:
{"supports":[
  {"name":"clutter level","spans":[{"text":"fewer distractions","score":0.98}]},
  {"name":"negative space","spans":[{"text":"strong negative space","score":0.99}]},
  {"name":"color temperature","spans":[{"text":"warm light","score":0.97}]},
  {"name":"time of day","spans":[{"text":"at night","score":0.95}]}
]}

Example 2
USER REQUEST:
"Browse product shots and filter for front-facing, centered object, plain background, no hands."
ATTRIBUTES:
[{"name":"viewpoint angle","type":"continuous"},{"name":"object centering","type":"continuous"},{"name":"background complexity","type":"ordinal"},{"name":"hand presence","type":"categorical"}]
OUTPUT:
{"supports":[
  {"name":"viewpoint angle","spans":[{"text":"front-facing","score":0.98}]},
  {"name":"object centering","spans":[{"text":"centered object","score":0.99}]},
  {"name":"background complexity","spans":[{"text":"plain background","score":0.97}]},
  {"name":"hand presence","spans":[{"text":"no hands","score":0.96}]}
]}
""".strip()

ATTRIBUTE_SUPPORT_USER_PROMPT_TEMPLATE = """
USER REQUEST:
{prompt}

ATTRIBUTES:
{attributes_json}
""".strip()

ATTRIBUTE_VALUE_SUGGESTION_SYSTEM_PROMPT = """
You are designing label sets for zero-shot image classification.

Task:
Given one attribute and its type, propose 2 to 9 short values that are useful
for scoring image diversity.

Output rules (strict):
- Return ONLY valid JSON.
- Format exactly:
  {"values": ["...", "...", "..."]}.
- Keep values short, concrete, and image-grounded.
- No explanations, no markdown, no extra keys.

Value rules:
- If type is categorical:
  - return distinct unordered classes
  - cover the main visually meaningful modes of the attribute
  - avoid overlap or near-synonyms
- If type is ordinal:
  - return ordered stages from low to high
  - use a progression with clear visual separation
  - prefer 3 to 7 stages unless the request explicitly asks for more
- Avoid vague labels like "other", "mixed", "normal", "varied" unless truly necessary.
- Prefer labels that CLIP-style text prompts can map to visible image content.

Example 1
ATTRIBUTE:
"lesion type" (categorical), 4 values
OUTPUT:
{"values":["melanoma","nevus","basal cell carcinoma","seborrheic keratosis"]}

Example 2
ATTRIBUTE:
"severity" (ordinal), 4 values
OUTPUT:
{"values":["none","mild","moderate","severe"]}

Example 3
ATTRIBUTE:
"background complexity" (ordinal), 5 values
OUTPUT:
{"values":["plain","simple","moderate","busy","cluttered"]}

Example 4
ATTRIBUTE:
"viewpoint" (categorical), 4 values
OUTPUT:
{"values":["front view","side view","top view","oblique view"]}
""".strip()

ATTRIBUTE_VALUE_SUGGESTION_USER_PROMPT_TEMPLATE = """
Attribute: {attribute} ({attribute_type})
Number of values: {n_values}
""".strip()

ATTRIBUTE_CONTINUOUS_ANCHORS_SYSTEM_PROMPT = """
You are defining two visual extremes for a continuous image attribute.

Task:
Given one continuous attribute, produce a low anchor and a high anchor that can
be embedded in CLIP text space to define a meaningful visual direction.

Output rules (strict):
- Return ONLY valid JSON.
- Format exactly:
  {"low": "...", "high": "..."}.
- Anchors must be concise visual descriptions.
- Anchors must describe opposite ends of the same attribute.
- Anchors must refer to plausible image content, not abstract numbers.
- No explanations, no markdown, no extra keys.

Anchor rules:
- Prefer image-like phrases such as "a photo of ...", "an image with ...", or
  similarly concrete visual descriptions.
- Make the contrast strong and visually legible.
- Avoid vague oppositions like "bad" vs "good" or "low" vs "high" without a
  visible manifestation.
- Keep each anchor roughly 3 to 10 words.

Example 1
ATTRIBUTE:
"age"
OUTPUT:
{"low":"a photo of a young person","high":"a photo of an elderly person"}

Example 2
ATTRIBUTE:
"brightness"
OUTPUT:
{"low":"a dark dimly lit image","high":"a bright strongly lit image"}

Example 3
ATTRIBUTE:
"blur"
OUTPUT:
{"low":"a sharp in-focus image","high":"a blurry out-of-focus image"}

Example 4
ATTRIBUTE:
"occlusion"
OUTPUT:
{"low":"an unobstructed fully visible object","high":"a heavily occluded object"}
""".strip()

ATTRIBUTE_CONTINUOUS_ANCHORS_USER_PROMPT_TEMPLATE = """
Attribute: {attribute}
""".strip()

# Axis builder prompt-ensemble configuration.
# This is the main on/off switch for Gemini-generated positive/negative text anchors:
# - 'llm': ask the same Gemini API path used for axis suggestion
# - 'template': use the old fixed prompt template
AXIS_BUILDER_AXIS_BOUNDS_TEXT_SOURCE = 'llm'  # 'template' | 'llm'

AXIS_BUILDER_USE_LLM_PROMPT_ENSEMBLE = True
AXIS_BUILDER_LLM_PROMPT_COUNT = 3

AXIS_BUILDER_PROMPT_ENSEMBLE_SYSTEM_PROMPT = """
You generate prompts for the two ends of a visual axis.

Task:
Given an ATTRIBUTE and a prompt count, infer the most natural visual opposite, then write prompts for both ends.

Important:
Do not assume the opposite is always "absence".
Use the most natural contrast.

Examples of natural contrasts:
- age -> old vs young
- blur -> blurry vs sharp
- brightness -> bright vs dark
- clutter -> cluttered vs sparse
- baroque -> baroque vs modern
- smile -> smiling vs neutral

Rules:
- `pos_prompts` = one end of the axis
- `neg_prompts` = the opposite end
- Both sides must match the same visual axis
- Use the provided dataset context when available so prompts stay plausible for the current collection
- Keep prompts short, concrete, and visual
- Use visible scenes, objects, faces, poses, or environments
- Avoid bad opposites like "no age" or "absence of baroque"

Output rules:
- Return ONLY valid JSON
- Format exactly:
  {"pos_prompts":["..."],"neg_prompts":["..."]}
- The number of strings in each list must exactly match the requested count
- No markdown
- No explanations
- No extra keys

Example
ATTRIBUTE: "age"
Requested prompt count: 3
OUTPUT:
{"pos_prompts":[
  "an elderly man with deep wrinkles",
  "an old woman with white hair",
  "an old tree with a thick twisted trunk"
],"neg_prompts":[
  "a young child with a smooth face",
  "a teenage girl with youthful features",
  "a young sapling with thin green branches"
]}
""".strip()
# AXIS_BUILDER_PROMPT_ENSEMBLE_SYSTEM_PROMPT = """
# You are a helpful assistant.

# Task:
# Produce examples of high-value and low-value manifestations of an attribute.

# Output rules (strict):
# - Return ONLY valid JSON.
# - Format exactly:
#   {"pos_prompts": ["...", "..."], "neg_prompts": ["...", "..."]}.
# - The number of strings in each list must exactly match the requested prompt count.
# - `pos_prompts` must describe strong or clear visual presence of the attribute.
# - `neg_prompts` must describe absence, minimality, neutrality, or an opposite visual expression of the attribute.
# - Prompts should vary context and subject.
# - Keep prompts short, concrete, and visually grounded.
# - If the attribute is abstract, express it through visible scenes, facial expressions, poses, objects, or environments.
# - No explanations, no markdown, no extra keys.

# Example 1
# ATTRIBUTE:
# "blur"
# Requested prompt count: 3
# OUTPUT:
# {"pos_prompts":[
#   "a blurry night street photo with motion blur",
#   "a slightly soft product photo",
#   "a heavily blurred surveillance frame"
# ],"neg_prompts":[
#   "a sharp portrait with crisp facial details",
#   "a clear product photo with hard edges",
#   "a landscape photo in perfect focus"
# ]}

# Example 2
# ATTRIBUTE:
# "age"
# Requested prompt count: 3
# OUTPUT:
# {"pos_prompts":[
#   "a portrait of a very old woman",
#   "an ancient stone ruin",
#   "a dinosaur fossil in a museum"
# ],"neg_prompts":[
#   "a newborn baby wrapped in a blanket",
#   "a brand new car in a showroom",
#   "a young kitten"
# ]}

# Example 3
# ATTRIBUTE:
# "baroque"
# Requested prompt count: 3
# OUTPUT:
# {"pos_prompts":[
#   "a baroque church interior",
#   "a baroque oil painting",
#   "a baroque marble sculpture"
# ],"neg_prompts":[
#   "a modern glass building",
#   "an abstract painting",
#   "a brutalist concrete sculpture"
# ]}

# Example 4
# ATTRIBUTE:
# "emotion"
# Requested prompt count: 3
# OUTPUT:
# {"pos_prompts":[
#   "a face showing intense joy",
#   "a person crying in deep sadness",
#   "a dramatic expression of anger"
# ],"neg_prompts":[
#   "a neutral face with no clear emotion",
#   "a blank expression",
#   "a calm passport-style portrait"
# ]}
# """.strip()

AXIS_BUILDER_PROMPT_ENSEMBLE_USER_PROMPT_TEMPLATE = """
ATTRIBUTE: {attribute}
Requested prompt count: {n_prompts}
OUTPUT:
""".strip()

# Prompt templates used to map text values into CLIP space.
ZERO_SHOT_VALUE_PROMPT_TEMPLATE = 'a photo where the {attribute} is {value}.'

ZERO_SHOT_VALUE_PROMPT_WITH_CONTEXT_TEMPLATE = 'a photo in the context of {context}, where the {attribute} is {value}.'
