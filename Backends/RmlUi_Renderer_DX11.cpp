#include "RmlUi_Renderer_DX11.h"
#include "RmlUi_Include_Windows.h"
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/DecorationTypes.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Geometry.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/MeshUtilities.h>
#include <RmlUi/Core/Platform.h>
#include <RmlUi/Core/SystemInterface.h>
#include <cmath>
#include <mutex>

// Helper for a common cleanup pattern in DX11
template <typename T>
static void Dx11CleanupResourceIfCreated(T*& resource)
{
	if (resource)
	{
		resource->Release();
		resource = nullptr;
	}
}

#ifndef RMLUI_PLATFORM_WIN32
	#error "Cannot build DX11 renderer on non-Windows platforms"
#endif // RMLUI_PLATFORM_WIN32

#include "d3dcompiler.h"

#ifdef RMLUI_DX_DEBUG
	#include <dxgidebug.h>
#endif

// Shader source code

#define MAX_NUM_STOPS 16
#define BLUR_SIZE 7
#define BLUR_NUM_WEIGHTS ((BLUR_SIZE + 1) / 2)

#define RMLUI_STRINGIFY_IMPL(x) #x
#define RMLUI_STRINGIFY(x) RMLUI_STRINGIFY_IMPL(x)

#define RMLUI_SHADER_HEADER "#define MAX_NUM_STOPS " RMLUI_STRINGIFY(MAX_NUM_STOPS) "\n#line " RMLUI_STRINGIFY(__LINE__) "\n"

#define RMLUI_SHADER_SAMPLING                                                                                                                       \
	"cbuffer RenderSampling : register(b2) { float4 rml_uv_transform; float4 rml_uv_limits; };\n"                                                   \
	"float2 RmlValidUv(float2 uv) { return clamp(uv, rml_uv_limits.xy, rml_uv_limits.zw); }\n"                                                      \
	"float RmlBorderWeight(float2 uv) { float2 texel = 2 * rml_uv_limits.xy; float2 a = saturate(1 + (uv - rml_uv_limits.xy) / texel); float2 b = " \
	"saturate(1 + (rml_uv_limits.zw - uv) / texel); return a.x * a.y * b.x * b.y; }\n"                                                              \
	"#define RmlSample(tex, samp, uv) (all(abs(rml_uv_limits.xy + rml_uv_limits.zw - 1) < 0.000001) ? tex.Sample(samp, uv) : tex.Sample(samp, "     \
	"RmlValidUv(uv)) * RmlBorderWeight(uv))\n"                                                                                                      \
	"#define RmlSampleLevel(tex, samp, uv, lod) (all(abs(rml_uv_limits.xy + rml_uv_limits.zw - 1) < 0.000001) ? tex.SampleLevel(samp, uv, lod) : "  \
	"tex.SampleLevel(samp, RmlValidUv(uv), lod) * RmlBorderWeight(uv))\n"

constexpr const char shader_vert_main[] = RMLUI_SHADER_HEADER R"(
struct VS_Input
{
    float2 position : POSITION;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

struct PS_INPUT
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

cbuffer SharedConstantBuffer : register(b0)
{
    float4x4 m_transform;
    float2 m_translate;
};

PS_INPUT VSMain(const VS_Input IN)
{
    PS_INPUT result = (PS_INPUT)0;

    float2 translatedPos = IN.position + m_translate;
    float4 resPos = mul(m_transform, float4(translatedPos.x, translatedPos.y, 0.0, 1.0));

    result.position = resPos;
    result.color = IN.color;
    result.uv = IN.uv;

    return result;
};
)";

constexpr const char shader_frag_texture[] = RMLUI_SHADER_HEADER R"(
struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

Texture2D g_InputTexture : register(t0);
SamplerState g_SamplerLinear : register(s0);

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    return IN.color * g_InputTexture.Sample(g_SamplerLinear, IN.uv);
};
)";

constexpr const char shader_frag_color[] = RMLUI_SHADER_HEADER R"(
struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    return IN.color;
};
)";

enum class ShaderGradientFunction { Linear, Radial, Conic, RepeatingLinear, RepeatingRadial, RepeatingConic }; // Must match shader definitions below.

// We need to round up at compile-time so that we can address each element of the float4s
#define CEILING(x, y) (((x) + (y) - 1) / (y))
static const char shader_frag_gradient[] = RMLUI_SHADER_HEADER "#define MAX_NUM_STOPS_PACKED (uint)" RMLUI_STRINGIFY(CEILING(MAX_NUM_STOPS, 4)) R"(
#define LINEAR 0
#define RADIAL 1
#define CONIC 2
#define REPEATING_LINEAR 3
#define REPEATING_RADIAL 4
#define REPEATING_CONIC 5
#define PI 3.14159265

cbuffer SharedConstantBuffer : register(b0)
{
    float4x4 m_transform;
    float2 m_translate;

    // One to one translation of the OpenGL uniforms results in a LOT of wasted space due to CBuffer alignment rules.
    // Changes from GL3:
    // - Moved m_num_stops below m_func (saved 4 bytes of padding).
    // - Packed m_stop_positions into a float4[MAX_NUM_STOPS / 4] array, as each array element starts a new 16-byte row.
    // The below layout has 0 bytes of padding.

    int m_func;   // one of the above definitions
    int m_num_stops;
    float2 m_p;   // linear: starting point,         radial: center,                        conic: center
    float2 m_v;   // linear: vector to ending point, radial: 2d curvature (inverse radius), conic: angled unit vector
    float4 m_stop_colors[MAX_NUM_STOPS];
    float4 m_stop_positions[MAX_NUM_STOPS_PACKED]; // normalized, 0 -> starting point, 1 -> ending point
};

// Hide the way the data is packed in the cbuffer through a macro
// @NOTE: Hardcoded for MAX_NUM_STOPS 16.
//        i >> 2 => i >> sqrt(MAX_NUM_STOPS)
#define GET_STOP_POS(i) (m_stop_positions[i >> 2][i & 3])

struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

#define glsl_mod(x,y) (((x)-(y)*floor((x)/(y))))

float4 lerp_stop_colors(float t) {
    float4 color = m_stop_colors[0];

    for (int i = 1; i < m_num_stops; i++)
        color = lerp(color, m_stop_colors[i], smoothstep(GET_STOP_POS(i-1), GET_STOP_POS(i), t));

    return color;
};

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    float t = 0.0;

    if (m_func == LINEAR || m_func == REPEATING_LINEAR) {
        float dist_square = dot(m_v, m_v);
        float2 V = IN.uv.xy - m_p;
        t = dot(m_v, V) / dist_square;
    }
    else if (m_func == RADIAL || m_func == REPEATING_RADIAL) {
        float2 V = IN.uv.xy - m_p;
        t = length(m_v * V);
    }
    else if (m_func == CONIC || m_func == REPEATING_CONIC) {
        float2x2 R = float2x2(m_v.x, -m_v.y, m_v.y, m_v.x);
        float2 V = mul((IN.uv.xy - m_p), R);
        t = 0.5 + atan2(-V.x, V.y) / (2.0 * PI);
    }

    if (m_func == REPEATING_LINEAR || m_func == REPEATING_RADIAL || m_func == REPEATING_CONIC) {
        float t0 = GET_STOP_POS(0);
        float t1 = GET_STOP_POS(m_num_stops - 1);
        t = t0 + glsl_mod(t - t0, t1 - t0);
    }

    return IN.color * lerp_stop_colors(t);
};
)";

// "Creation" by Danilo Guanabara, based on: https://www.shadertoy.com/view/XsXXDn
static const char shader_frag_creation[] = RMLUI_SHADER_HEADER R"(
struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

cbuffer SharedConstantBuffer : register(b0)
{
    float4x4 m_transform;
    float2 m_translate;
    float2 m_dimensions;
    float m_value;
};

#define glsl_mod(x,y) (((x)-(y)*floor((x)/(y))))

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    float t = m_value;
    float3 c;
    float l;
    for (int i = 0; i < 3; i++) {
        float2 p = IN.uv;
        float2 uv = p;
        p -= .5;
        p.x *= m_dimensions.x / m_dimensions.y;
        float z = t + ((float)i) * .07;
        l = length(p);
        uv += p / l * (sin(z) + 1.) * abs(sin(l * 9. - z - z));
        c[i] = .01 / length(glsl_mod(uv, 1.) - .5);
    }
    return float4(c / l, IN.color.a);
};
)";

constexpr const char shader_vert_passthrough[] = RMLUI_SHADER_HEADER RMLUI_SHADER_SAMPLING R"(
struct VS_Input
{
    float2 position : POSITION;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

cbuffer ConstantBuffer : register(b0)
{
    float4x4 m_transform;
    float2 m_translate;
};

PS_Input VSMain(const VS_Input IN)
{
    PS_Input result = (PS_Input)0;

    result.position = float4(IN.position.xy, 0.0f, 1.0f);
    result.uv = float2(IN.uv.x, 1.0f - IN.uv.y);
    if (any(rml_uv_transform.xy != 1) || any(rml_uv_transform.zw != 0))
        result.uv = result.uv * rml_uv_transform.xy + rml_uv_transform.zw;

    return result;
};
)";

constexpr const char shader_frag_passthrough[] = RMLUI_SHADER_HEADER RMLUI_SHADER_SAMPLING R"(
struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

Texture2D g_InputTexture : register(t0);
SamplerState g_SamplerLinear : register(s0);

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    return RmlSample(g_InputTexture, g_SamplerLinear, IN.uv);
};
)";

constexpr const char shader_frag_surface_backdrop[] = RMLUI_SHADER_HEADER RMLUI_SHADER_SAMPLING R"(
struct PS_Input { float4 position : SV_Position; float4 color : COLOR; float2 uv : TEXCOORD; };
Texture2D foreground : register(t0);
Texture2D background : register(t1);
SamplerState linearClamp : register(s0);
cbuffer SurfaceMapping : register(b1) { float4 m_backdrop_mapping; };
float4 PSMain(PS_Input input) : SV_TARGET {
    float4 color = RmlSample(foreground, linearClamp, input.uv);
    const float2 uv = (input.position.xy + m_backdrop_mapping.xy) * m_backdrop_mapping.zw;
    return color + background.Sample(linearClamp, uv) * (1.0 - color.a);
}
)";

static const char shader_frag_foreground_resolve[] = RMLUI_SHADER_HEADER R"(
Texture2DMS<float4, 4> background : register(t0);
Texture2DMS<float4, 4> foreground : register(t1);
float4 PSMain(float4 position : SV_Position) : SV_TARGET {
    const int2 pixel = int2(position.xy);
    float4 result = 0;
    [unroll] for (int sample = 0; sample < 4; ++sample) {
        const float4 f = foreground.Load(pixel, sample);
        const float4 color = f + background.Load(pixel, sample) * (1.0 - f.a);
        result += color;
    }
    return result * 0.25;
}
)";

static const char shader_frag_resolve[] = RMLUI_SHADER_HEADER R"(
Texture2DMS<float4, 4> g_InputTexture : register(t0);

float4 PSMain(float4 position : SV_Position) : SV_TARGET
{
    const int2 pixel = int2(position.xy);
    return (g_InputTexture.Load(pixel, 0) + g_InputTexture.Load(pixel, 1)
        + g_InputTexture.Load(pixel, 2) + g_InputTexture.Load(pixel, 3)) * 0.25f;
};
)";

// Same MSAA resolve, with integer addressing for a cropped working target.
static const char shader_frag_resolve_region[] = RMLUI_SHADER_HEADER R"(
Texture2DMS<float4, 4> g_InputTexture : register(t0);
cbuffer ResolveRegion : register(b1) { int2 source_offset; };
float4 PSMain(float4 position : SV_Position) : SV_TARGET
{
    const int2 pixel = int2(position.xy) + source_offset;
    return (g_InputTexture.Load(pixel, 0) + g_InputTexture.Load(pixel, 1)
        + g_InputTexture.Load(pixel, 2) + g_InputTexture.Load(pixel, 3)) * 0.25f;
};
)";

static const char shader_frag_color_matrix[] = RMLUI_SHADER_HEADER RMLUI_SHADER_SAMPLING R"(

Texture2D g_InputTexture : register(t0);
SamplerState g_SamplerLinear : register(s0);

cbuffer ConstantBuffer : register(b1)
{
    float4x4 m_color_matrix;
};

struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    // The general case uses a 4x5 color matrix for full rgba transformation, plus a constant term with the last column.
    // However, we only consider the case of rgb transformations. Thus, we could in principle use a 3x4 matrix, but we
    // keep the alpha row for simplicity.
    // In the general case we should do the matrix transformation in non-premultiplied space. However, without alpha
    // transformations, we can do it directly in premultiplied space to avoid the extra division and multiplication
    // steps. In this space, the constant term needs to be multiplied by the alpha value, instead of unity.
    float4 texColor = RmlSample(g_InputTexture, g_SamplerLinear, IN.uv);
    float3 transformedColor = mul(m_color_matrix, texColor).rgb;
    return float4(transformedColor, texColor.a);
};
)";

static const char shader_frag_blend_mask[] = RMLUI_SHADER_HEADER RMLUI_SHADER_SAMPLING R"(
Texture2D g_InputTexture : register(t0);
Texture2D g_MaskTexture : register(t1);
SamplerState g_SamplerLinear : register(s0);
cbuffer MaskMapping : register(b1) { float4 m_mask_mapping; float4 m_mask_limit; };

struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    float4 texColor = RmlSample(g_InputTexture, g_SamplerLinear, IN.uv);
    float2 uv=(IN.position.xy+m_mask_mapping.xy)*m_mask_mapping.zw;
    float maskAlpha = all(uv>=0) && all(uv<m_mask_limit.xy) ? g_MaskTexture.Sample(g_SamplerLinear,uv).a : 0;
    return texColor * maskAlpha;
};
)";

#define RMLUI_SHADER_BLUR_HEADER              \
	RMLUI_SHADER_HEADER RMLUI_SHADER_SAMPLING \
		"\n#define BLUR_SIZE " RMLUI_STRINGIFY(BLUR_SIZE) "\n#define BLUR_NUM_WEIGHTS " RMLUI_STRINGIFY(BLUR_NUM_WEIGHTS)

static const char shader_vert_blur[] = RMLUI_SHADER_BLUR_HEADER R"(
struct VS_Input
{
    float2 position : POSITION;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

struct PS_INPUT
{
    float4 position : SV_Position;
    float2 uv[BLUR_SIZE] : TEXCOORD;
};

cbuffer SharedConstantBuffer : register(b0)
{
    float4x4 m_transform;
    float2 m_translate;
    float2 m_texelOffset;
    float4 m_weights;
    float2 m_texCoordMin;
    float2 m_texCoordMax;
};

PS_INPUT VSMain(const VS_Input IN)
{
    PS_INPUT result = (PS_INPUT)0;

    float2 uv = IN.uv;
    if (any(rml_uv_transform.xy != 1))
        uv = float2(IN.uv.x * rml_uv_transform.x, 1.0f - rml_uv_transform.y + IN.uv.y * rml_uv_transform.y);
    for (int i = 0; i < BLUR_SIZE; i++) {
        result.uv[i] = uv - float(i - BLUR_NUM_WEIGHTS + 1) * m_texelOffset;
    }
    result.position = float4(IN.position.xy, 1.0, 1.0);

    return result;
};
)";

static const char shader_frag_blur[] = RMLUI_SHADER_BLUR_HEADER R"(
Texture2D g_InputTexture : register(t0);
SamplerState g_SamplerLinear : register(s0);

cbuffer SharedConstantBuffer : register(b0)
{
    float4x4 m_transform;
    float2 m_translate;
    float2 m_texelOffset;
    float4 m_weights;
    float2 m_texCoordMin;
    float2 m_texCoordMax;
};

struct PS_Input
{
    float4 position : SV_Position;
    float2 uv[BLUR_SIZE] : TEXCOORD;
};

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    for(int i = 0; i < BLUR_SIZE; i++)
    {
        float2 in_region = step(m_texCoordMin, IN.uv[i]) * step(IN.uv[i], m_texCoordMax);
        color += RmlSample(g_InputTexture, g_SamplerLinear, float2(IN.uv[i].x, 1.0f - IN.uv[i].y)) * in_region.x * in_region.y * m_weights[abs(i - BLUR_NUM_WEIGHTS + 1)];
    }
    return color;
};
)";

static const char shader_frag_blur_paired[] = RMLUI_SHADER_BLUR_HEADER R"(
Texture2D g_InputTexture : register(t0);
SamplerState g_SamplerLinear : register(s0);

cbuffer SharedConstantBuffer : register(b0)
{
    float4x4 m_transform;
    float2 m_translate;
    float2 m_texelOffset;
    float4 m_weights;
    float2 m_texCoordMin;
    float2 m_texCoordMax;
    float4 m_pairedWeightsOffsets;
};

struct PS_Input
{
    float4 position : SV_Position;
    float2 uv[BLUR_SIZE] : TEXCOORD;
};

float4 SampleBlur(float2 uv)
{
    return RmlSampleLevel(g_InputTexture, g_SamplerLinear, float2(uv.x, 1.0f - uv.y), 0);
}

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    // The vertex shader supplies both axes. The PS texel offset still describes the vertical pass during horizontal blur.
    const bool interior = all(IN.uv[0] >= m_texCoordMin) && all(IN.uv[0] <= m_texCoordMax)
        && all(IN.uv[BLUR_SIZE - 1] >= m_texCoordMin) && all(IN.uv[BLUR_SIZE - 1] <= m_texCoordMax);
    [branch] if (interior)
    {
        const float inner_weight = m_pairedWeightsOffsets.x;
        const float outer_weight = m_pairedWeightsOffsets.y;
        const float inner_offset = m_pairedWeightsOffsets.z;
        const float outer_offset = m_pairedWeightsOffsets.w;
        float4 color = (SampleBlur(lerp(IN.uv[3], IN.uv[2], inner_offset))
            + SampleBlur(lerp(IN.uv[3], IN.uv[4], inner_offset))) * inner_weight;
        color += (SampleBlur(lerp(IN.uv[1], IN.uv[0], outer_offset))
            + SampleBlur(lerp(IN.uv[5], IN.uv[6], outer_offset))) * outer_weight;
        return color;
    }

    // A bilinear pair straddling the filter bounds would blend a tap which the reference kernel rejects.
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < BLUR_SIZE; i++)
    {
        float2 in_region = step(m_texCoordMin, IN.uv[i]) * step(IN.uv[i], m_texCoordMax);
        color += SampleBlur(IN.uv[i]) * in_region.x * in_region.y * m_weights[abs(i - BLUR_NUM_WEIGHTS + 1)];
    }
    return color;
};
)";

static const char shader_frag_drop_shadow[] = RMLUI_SHADER_HEADER RMLUI_SHADER_SAMPLING R"(
Texture2D g_InputTexture : register(t0);
SamplerState g_SamplerLinear : register(s0);

cbuffer DropShadowBuffer : register(b1)
{
    float2 m_texCoordMin;
    float2 m_texCoordMax;
    float4 m_color;
};

struct PS_Input
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

float4 PSMain(const PS_Input IN) : SV_TARGET
{
    float2 in_region = step(m_texCoordMin, IN.uv) * step(IN.uv, m_texCoordMax);
    return RmlSample(g_InputTexture, g_SamplerLinear, IN.uv).a * in_region.x * in_region.y * m_color;
};
)";

enum class ProgramId {
	None,
	Color,
	Texture,
	Gradient,
	Creation,
	Passthrough,
	ColorMatrix,
	BlendMask,
	Blur,
	DropShadow,
	Resolve,
	ResolveRegion,
	ForegroundResolve,
	BlurPaired,
	SurfaceBackdrop,
	Count,
};
enum class VertShaderId {
	Main,
	Passthrough,
	Blur,
	Count,
};
enum class FragShaderId {
	SurfaceBackdrop,
	Color,
	Texture,
	Gradient,
	Creation,
	Passthrough,
	ColorMatrix,
	BlendMask,
	Blur,
	DropShadow,
	Count,
};
enum class ShaderType {
	Vertex,
	Fragment,
};

namespace Gfx {

struct VertShaderDefinition {
	VertShaderId id;
	const char* name_str;
	const char* code_str;
	const size_t code_size;
};
struct FragShaderDefinition {
	FragShaderId id;
	const char* name_str;
	const char* code_str;
	const size_t code_size;
};
struct ProgramDefinition {
	ProgramId id;
	const char* name_str;
	VertShaderId vert_shader;
	FragShaderId frag_shader;
};

// clang-format off
static const VertShaderDefinition vert_shader_definitions[] = {
    {VertShaderId::Main,        "main",         shader_vert_main,           sizeof(shader_vert_main)},
    {VertShaderId::Passthrough, "passthrough",  shader_vert_passthrough,    sizeof(shader_vert_passthrough)},
    {VertShaderId::Blur,        "blur",         shader_vert_blur,           sizeof(shader_vert_blur)},
};
static const FragShaderDefinition frag_shader_definitions[] = {
    {FragShaderId::SurfaceBackdrop, "surface_backdrop", shader_frag_surface_backdrop, sizeof(shader_frag_surface_backdrop)},
    {FragShaderId::Color,       "color",        shader_frag_color,          sizeof(shader_frag_color)},
    {FragShaderId::Texture,     "texture",      shader_frag_texture,        sizeof(shader_frag_texture)},
    {FragShaderId::Gradient,    "gradient",     shader_frag_gradient,       sizeof(shader_frag_gradient)},
    {FragShaderId::Creation,    "creation",     shader_frag_creation,       sizeof(shader_frag_creation)},
    {FragShaderId::Passthrough, "passthrough",  shader_frag_passthrough,    sizeof(shader_frag_passthrough)},
    {FragShaderId::ColorMatrix, "color_matrix", shader_frag_color_matrix,   sizeof(shader_frag_color_matrix)},
    {FragShaderId::BlendMask,   "blend_mask",   shader_frag_blend_mask,     sizeof(shader_frag_blend_mask)},
    {FragShaderId::Blur,        "blur",         shader_frag_blur,           sizeof(shader_frag_blur)},
    {FragShaderId::DropShadow,  "drop_shadow",  shader_frag_drop_shadow,    sizeof(shader_frag_drop_shadow)},
};
static const ProgramDefinition program_definitions[] = {
    {ProgramId::SurfaceBackdrop, "surface_backdrop", VertShaderId::Passthrough, FragShaderId::SurfaceBackdrop},
    {ProgramId::Color,       "color",        VertShaderId::Main,        FragShaderId::Color},
    {ProgramId::Texture,     "texture",      VertShaderId::Main,        FragShaderId::Texture},
    {ProgramId::Gradient,    "gradient",     VertShaderId::Main,        FragShaderId::Gradient},
    {ProgramId::Creation,    "creation",     VertShaderId::Main,        FragShaderId::Creation},
    {ProgramId::Passthrough, "passthrough",  VertShaderId::Passthrough, FragShaderId::Passthrough},
    {ProgramId::ColorMatrix, "color_matrix", VertShaderId::Passthrough, FragShaderId::ColorMatrix},
    {ProgramId::BlendMask,   "blend_mask",   VertShaderId::Passthrough, FragShaderId::BlendMask},
    {ProgramId::Blur,        "blur",         VertShaderId::Blur,        FragShaderId::Blur},
    {ProgramId::DropShadow,  "drop_shadow",  VertShaderId::Passthrough, FragShaderId::DropShadow},
};
// clang-format on

template <typename T, typename Enum>
class EnumArray {
public:
	const T& operator[](Enum id) const
	{
		RMLUI_ASSERT((size_t)id < (size_t)Enum::Count);
		return ids[size_t(id)];
	}
	T& operator[](Enum id)
	{
		RMLUI_ASSERT((size_t)id < (size_t)Enum::Count);
		return ids[size_t(id)];
	}
	auto begin() const { return ids.begin(); }
	auto end() const { return ids.end(); }

private:
	Rml::Array<T, (size_t)Enum::Count> ids = {};
};

struct ShaderProgram {
public:
	ID3D11VertexShader* vertex_shader;
	ID3D11PixelShader* pixel_shader;
};

using Programs = EnumArray<ShaderProgram, ProgramId>;
using VertShaders = EnumArray<ID3DBlob*, VertShaderId>;
using FragShaders = EnumArray<ID3DBlob*, FragShaderId>;

struct ProgramData {
	Programs programs;
	VertShaders vert_shaders;
	FragShaders frag_shaders;
};

// Create the shader, 'shader_type' is either GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
static bool CreateShader(ID3DBlob*& out_shader_dxil, ShaderType shader_type, const char* code_string, const size_t code_length,
	bool optional = false)
{
	RMLUI_ASSERT(shader_type == ShaderType::Vertex || shader_type == ShaderType::Fragment);

#ifdef RMLUI_DX_DEBUG
	static const UINT default_shader_flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
	static const UINT default_shader_flags = 0;
#endif

	// Compile shader
	ID3DBlob* p_error_buff = nullptr;

	HRESULT result = D3DCompile(code_string, code_length, nullptr, nullptr, nullptr, shader_type == ShaderType::Vertex ? "VSMain" : "PSMain",
		shader_type == ShaderType::Vertex ? "vs_5_0" : "ps_5_0", default_shader_flags, 0, &out_shader_dxil, &p_error_buff);
	if (!optional)
		RMLUI_DX_ASSERTMSG(result, "failed to D3DCompile");
	if (FAILED(result))
	{
		Rml::Log::Message(optional ? Rml::Log::LT_WARNING : Rml::Log::LT_ERROR, "Failed to compile %sDX11 shader: %s",
			optional ? "optional " : "", p_error_buff ? (const char*)p_error_buff->GetBufferPointer() : "no compiler diagnostics");
		Dx11CleanupResourceIfCreated(p_error_buff);
		Dx11CleanupResourceIfCreated(out_shader_dxil);
		return false;
	}

	Dx11CleanupResourceIfCreated(p_error_buff);

	return true;
}

static bool CreateProgram(ID3D11Device* p_device, ShaderProgram& out_program, ID3DBlob* vertex_shader, ID3DBlob* fragment_shader,
	bool optional = false)
{
	RMLUI_ASSERT(vertex_shader);
	RMLUI_ASSERT(fragment_shader);

	HRESULT result =
		p_device->CreateVertexShader(vertex_shader->GetBufferPointer(), vertex_shader->GetBufferSize(), nullptr, &out_program.vertex_shader);
	if (!optional)
		RMLUI_DX_ASSERTMSG(result, "failed to CreateVertexShader");
	if (FAILED(result))
	{
#ifdef RMLUI_DX_DEBUG
		Rml::Log::Message(Rml::Log::Type::LT_ERROR, "failed to create vertex shader: %d", result);
#endif
		return false;
	}

	result = p_device->CreatePixelShader(fragment_shader->GetBufferPointer(), fragment_shader->GetBufferSize(), nullptr, &out_program.pixel_shader);
	if (!optional)
		RMLUI_DX_ASSERTMSG(result, "failed to CreatePixelShader");
	if (FAILED(result))
	{
#ifdef RMLUI_DX_DEBUG
		Rml::Log::Message(Rml::Log::Type::LT_ERROR, "failed to create pixel shader: %d", result);
#endif
		return false;
	}

	return true;
}

struct StencilState {
	uint64_t generation = 0;
	Rml::Vector2i origin = {}, size = {};
};

struct RenderTargetData {
	Rml::Rectanglei bounds = Rml::Rectanglei::MakeInvalid();
	Rml::SharedPtr<StencilState> stencil;
	int width = 0, height = 0;
	UINT sample_count = 1;
	ID3D11RenderTargetView* render_target_view = nullptr;                   // To write to colour attachment buffer
	ID3D11DepthStencilView* depth_stencil_view = nullptr;                   // To write to stencil buffer
	ID3D11Texture2D* render_target_texture = nullptr;                       // For MSAA resolve
	ID3D11ShaderResourceView* render_target_shader_resource_view = nullptr; // To blit
	bool owns_depth_stencil_buffer = false;
};

enum class RenderTargetAttachment { None, Depth, DepthStencil };

static bool CreateRenderTarget(ID3D11Device* p_device, RenderTargetData& out_rt, int width, int height, int samples,
	RenderTargetAttachment attachment, ID3D11DepthStencilView* shared_depth_stencil_buffer)
{
	// Generate render target
	D3D11_TEXTURE2D_DESC texture_desc = {};
	ZeroMemory(&texture_desc, sizeof(D3D11_TEXTURE2D_DESC));
	texture_desc.Width = width;
	texture_desc.Height = height;
	texture_desc.MipLevels = 1;
	texture_desc.ArraySize = 1;
	texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	texture_desc.SampleDesc.Count = (samples > 1) ? samples : 1; // MSAA
	texture_desc.SampleDesc.Quality = (samples > 1) ? D3D11_STANDARD_MULTISAMPLE_PATTERN : 0;
	texture_desc.Usage = D3D11_USAGE_DEFAULT;
	texture_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	texture_desc.CPUAccessFlags = 0;
	texture_desc.MiscFlags = 0;

	ID3D11Texture2D* rt_texture = nullptr;
	HRESULT result = p_device->CreateTexture2D(&texture_desc, nullptr, &rt_texture);
	RMLUI_DX_ASSERTMSG(result, "failed to CreateTexture2D");
	if (FAILED(result))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateTexture2D (%d)", result);
		return false;
	}

	D3D11_RENDER_TARGET_VIEW_DESC render_target_view_desc = {};
	render_target_view_desc.Format = texture_desc.Format;
	render_target_view_desc.ViewDimension = samples > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
	render_target_view_desc.Texture2D.MipSlice = 0;

	ID3D11RenderTargetView* render_target_view = nullptr;
	result = p_device->CreateRenderTargetView(rt_texture, &render_target_view_desc, &render_target_view);
	RMLUI_DX_ASSERTMSG(result, "failed to CreateRenderTargetView");
	if (FAILED(result))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateRenderTargetView (%d)", result);
		return false;
	}

	D3D11_SHADER_RESOURCE_VIEW_DESC render_target_shader_resource_view_desc = {};
	render_target_shader_resource_view_desc.Format = texture_desc.Format;
	render_target_shader_resource_view_desc.ViewDimension = samples > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
	render_target_shader_resource_view_desc.Texture2D.MostDetailedMip = 0;
	render_target_shader_resource_view_desc.Texture2D.MipLevels = 1;

	ID3D11ShaderResourceView* render_target_shader_resource_view = nullptr;
	result = p_device->CreateShaderResourceView(rt_texture, &render_target_shader_resource_view_desc, &render_target_shader_resource_view);
	RMLUI_DX_ASSERTMSG(result, "failed to CreateShaderResourceView");
	if (FAILED(result))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateShaderResourceView (%d)", result);
		return false;
	}

	// Generate stencil buffer if necessary
	ID3D11DepthStencilView* depth_stencil_view = nullptr;

	if (attachment != RenderTargetAttachment::None)
	{
		if (shared_depth_stencil_buffer)
		{
			// Share the depth/stencil buffer
			depth_stencil_view = shared_depth_stencil_buffer;
			depth_stencil_view->AddRef(); // Increase reference count since we're sharing it
		}
		else
		{
			// Create a new depth/stencil buffer
			D3D11_TEXTURE2D_DESC depth_desc = {};
			depth_desc.Width = width;
			depth_desc.Height = height;
			depth_desc.MipLevels = 1;
			depth_desc.ArraySize = 1;
			depth_desc.Format =
				(attachment == RenderTargetAttachment::DepthStencil) ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : DXGI_FORMAT_D24_UNORM_S8_UINT;
			depth_desc.SampleDesc = texture_desc.SampleDesc;
			depth_desc.Usage = D3D11_USAGE_DEFAULT;
			depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
			depth_desc.CPUAccessFlags = 0;
			depth_desc.MiscFlags = 0;

			ID3D11Texture2D* depth_stencil_texture = nullptr;
			result = p_device->CreateTexture2D(&depth_desc, nullptr, &depth_stencil_texture);
			RMLUI_DX_ASSERTMSG(result, "failed to CreateTexture2D");
			if (FAILED(result))
			{
				Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateTexture2D (%d)", result);
				return false;
			}

			D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
			dsvDesc.Format = depth_desc.Format;
			dsvDesc.ViewDimension = (samples > 0) ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
			dsvDesc.Texture2D.MipSlice = 0;

			result = p_device->CreateDepthStencilView(depth_stencil_texture, &dsvDesc, &depth_stencil_view);
			RMLUI_DX_ASSERTMSG(result, "failed to CreateDepthStencilView");
			if (FAILED(result))
			{
				depth_stencil_texture->Release();
				Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateDepthStencilView (%d)", result);
				return false;
			}

			depth_stencil_texture->Release();
		}
	}

	out_rt = {};
	out_rt.bounds = Rml::Rectanglei::FromSize({width, height});
	out_rt.stencil = depth_stencil_view ? Rml::MakeShared<StencilState>() : nullptr;
	out_rt.width = width;
	out_rt.height = height;
	out_rt.sample_count = texture_desc.SampleDesc.Count;
	out_rt.render_target_view = render_target_view;
	out_rt.render_target_texture = rt_texture;
	out_rt.render_target_shader_resource_view = render_target_shader_resource_view;
	out_rt.depth_stencil_view = depth_stencil_view;
	out_rt.owns_depth_stencil_buffer = shared_depth_stencil_buffer != nullptr;

	return true;
}

static void DestroyRenderTarget(RenderTargetData& rt)
{
	Dx11CleanupResourceIfCreated(rt.render_target_view);
	Dx11CleanupResourceIfCreated(rt.render_target_texture);
	Dx11CleanupResourceIfCreated(rt.render_target_shader_resource_view);
	Dx11CleanupResourceIfCreated(rt.depth_stencil_view);
	rt = {};
}

static inline void BindTexture(ID3D11DeviceContext* context, const RenderTargetData& rt, uint32_t slot = 0)
{
	context->PSSetShaderResources(slot, 1, &rt.render_target_shader_resource_view);
}

static inline void BindRenderTarget(ID3D11DeviceContext* context, const RenderTargetData& rt)
{
	context->OMSetRenderTargets(1, &rt.render_target_view, rt.depth_stencil_view);
}

static bool CreateShaders(ID3D11Device* p_device, ProgramData& data)
{
	RMLUI_ASSERT(std::all_of(data.vert_shaders.begin(), data.vert_shaders.end(), [](auto&& value) { return value == nullptr; }));
	RMLUI_ASSERT(std::all_of(data.frag_shaders.begin(), data.frag_shaders.end(), [](auto&& value) { return value == nullptr; }));
	RMLUI_ASSERT(std::all_of(data.programs.begin(), data.programs.end(),
		[](const ShaderProgram& value) { return value.vertex_shader == nullptr || value.pixel_shader == nullptr; }));
	auto ReportError = [](const char* type, const char* name) {
		Rml::Log::Message(Rml::Log::LT_ERROR, "Could not create shader %s: '%s'.", type, name);
		return false;
	};

	for (const VertShaderDefinition& def : vert_shader_definitions)
	{
		if (!CreateShader(data.vert_shaders[def.id], ShaderType::Vertex, def.code_str, def.code_size))
			return ReportError("vertex shader", def.name_str);
	}

	for (const FragShaderDefinition& def : frag_shader_definitions)
	{
		if (!CreateShader(data.frag_shaders[def.id], ShaderType::Fragment, def.code_str, def.code_size))
			return ReportError("fragment shader", def.name_str);
	}

	for (const ProgramDefinition& def : program_definitions)
	{
		if (!CreateProgram(p_device, data.programs[def.id], data.vert_shaders[def.vert_shader], data.frag_shaders[def.frag_shader]))
			return ReportError("program", def.name_str);
	}

	return true;
}

static void CreateOptionalProgram(ID3D11Device* device, ProgramData& data, ProgramId program, VertShaderId vertex,
	const char* fragment_source, size_t source_size)
{
	ID3DBlob* fragment = nullptr;
	ShaderProgram candidate{};
	const bool compiled = CreateShader(fragment, ShaderType::Fragment, fragment_source, source_size, true);
	if (compiled && CreateProgram(device, candidate, data.vert_shaders[vertex], fragment, true))
		data.programs[program] = candidate;
	else
	{
		if (compiled)
			Rml::Log::Message(Rml::Log::LT_WARNING, "Could not create optional DX11 program %d; using the reference path.", int(program));
		Dx11CleanupResourceIfCreated(candidate.vertex_shader);
		Dx11CleanupResourceIfCreated(candidate.pixel_shader);
	}
	Dx11CleanupResourceIfCreated(fragment);
}

static void DestroyShaders(const ProgramData& data)
{
	for (ShaderProgram programs : data.programs)
	{
		Dx11CleanupResourceIfCreated(programs.vertex_shader);
		Dx11CleanupResourceIfCreated(programs.pixel_shader);
	}

	for (ID3DBlob* shader_blob : data.vert_shaders)
	{
		Dx11CleanupResourceIfCreated(shader_blob);
	}

	for (ID3DBlob* shader_blob : data.frag_shaders)
	{
		Dx11CleanupResourceIfCreated(shader_blob);
	}
}

} // namespace Gfx

RenderInterface_DX11::RenderInterface_DX11(ID3D11Device* p_d3d_device)
{
	if (!p_d3d_device || p_d3d_device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0) return;

	// Assign D3D resources
	m_d3d_device = p_d3d_device;
	m_d3d_device->AddRef();
	if (FAILED(m_d3d_device->QueryInterface(__uuidof(ID3D11Device1), (void**)&m_d3d_device_1))) return;
	m_d3d_device_1->GetImmediateContext1(&m_d3d_context);
	if (!m_d3d_context) return;
	m_render_layers->SetD3DResources(m_d3d_device);
	m_surface_render_layers.SetD3DResources(m_d3d_device);

	// Pre-cache quad for blitting
	Rml::Mesh mesh;
	Rml::MeshUtilities::GenerateQuad(mesh, Rml::Vector2f(-1), Rml::Vector2f(2), {});
	m_fullscreen_quad_geometry = RenderInterface_DX11::CompileGeometry(mesh.vertices, mesh.indices);
	if (!m_fullscreen_quad_geometry) return;

	D3D11_BUFFER_DESC quad_desc{};
	quad_desc.ByteWidth = 4 * sizeof(Rml::Vertex);
	quad_desc.Usage = D3D11_USAGE_DYNAMIC;
	quad_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	quad_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	if (m_fullscreen_quad_geometry && SUCCEEDED(m_d3d_device->CreateBuffer(&quad_desc, nullptr, &m_dynamic_quad_geometry.vertex_buffer)))
	{
		const auto* fullscreen = reinterpret_cast<const DX11_GeometryData*>(m_fullscreen_quad_geometry);
		m_dynamic_quad_geometry.index_buffer = fullscreen->index_buffer;
		m_dynamic_quad_geometry.index_buffer->AddRef();
		m_dynamic_quad_geometry.index_count = fullscreen->index_count;
	}
	else
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create the dynamic quad buffer");


	if (!m_blend_state_enable)
	{
		// RmlUi serves vertex colors and textures with premultiplied alpha, set the blend mode accordingly.
		// Equivalent to glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA).
		D3D11_BLEND_DESC blend_desc{};
		ZeroMemory(&blend_desc, sizeof(blend_desc));
		blend_desc.AlphaToCoverageEnable = false;
		blend_desc.IndependentBlendEnable = false;
		blend_desc.RenderTarget[0].BlendEnable = true;
		blend_desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
		blend_desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
		blend_desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
		blend_desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
		blend_desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
		blend_desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
		blend_desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
		HRESULT result = m_d3d_device->CreateBlendState(&blend_desc, &m_blend_state_enable);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBlendState");
		if (FAILED(result))
		{
			Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateBlendState (%d)", result);
			return;
		}

		ZeroMemory(&blend_desc, sizeof(blend_desc));
		blend_desc.RenderTarget[0].RenderTargetWriteMask = 0;
		result = m_d3d_device->CreateBlendState(&blend_desc, &m_blend_state_disable_color);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBlendState");
		if (FAILED(result))
		{
			Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateBlendState (%d)", result);
			return;
		}

		blend_desc.RenderTarget[0].BlendEnable = true;
		blend_desc.RenderTarget[0].SrcBlend = D3D11_BLEND_BLEND_FACTOR;
		blend_desc.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
		blend_desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
		blend_desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_BLEND_FACTOR;
		blend_desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
		blend_desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
		blend_desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
		result = m_d3d_device->CreateBlendState(&blend_desc, &m_blend_state_color_filter);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBlendState");
		if (FAILED(result))
		{
			Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateBlendState (%d)", result);
			return;
		}
	}

	// Scissor regions require a rasterizer state. Cache one for scissor on and off
	{
		D3D11_RASTERIZER_DESC rasterizerDesc{};
		rasterizerDesc.FillMode = D3D11_FILL_SOLID;
		rasterizerDesc.CullMode = D3D11_CULL_NONE;
		rasterizerDesc.FrontCounterClockwise = false;
		rasterizerDesc.DepthBias = D3D11_DEFAULT_DEPTH_BIAS;
		rasterizerDesc.SlopeScaledDepthBias = D3D11_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
		rasterizerDesc.DepthBiasClamp = D3D11_DEFAULT_DEPTH_BIAS_CLAMP;
		rasterizerDesc.DepthClipEnable = false;
		rasterizerDesc.ScissorEnable = true;
		rasterizerDesc.MultisampleEnable = MsaaSamples > 1;
		rasterizerDesc.AntialiasedLineEnable = MsaaSamples > 1;

		HRESULT result = m_d3d_device->CreateRasterizerState(&rasterizerDesc, &m_rasterizer_state_scissor_enabled);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateRasterizerState");
		if (FAILED(result))
		{
			Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateRasterizerState (scissor: enabled) (%d)", result);
			return;
		}

		rasterizerDesc.ScissorEnable = false;

		result = m_d3d_device->CreateRasterizerState(&rasterizerDesc, &m_rasterizer_state_scissor_disabled);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateRasterizerState");
		if (FAILED(result))
		{
			Rml::Log::Message(Rml::Log::LT_ERROR, "ID3D11Device::CreateRasterizerState (scissor: disabled) (%d)", result);
			return;
		}
	}

	// Programs are immutable and device-owned. Helper renderers need their own state and buffers,
	// but must not compile the same HLSL again on the first shadow or after a viewport change.
	{
		static std::mutex programs_mutex;
		static Rml::WeakPtr<const Gfx::ProgramData> cached_programs;
		static ID3D11Device* cached_device = nullptr;
		std::lock_guard<std::mutex> lock(programs_mutex);
		if (cached_device == m_d3d_device) program_data = cached_programs.lock();
		if (!program_data)
		{
			Rml::SharedPtr<Gfx::ProgramData> programs(new Gfx::ProgramData(), [](Gfx::ProgramData* data)
			{
				Gfx::DestroyShaders(*data);
				delete data;
			});
			if (!Gfx::CreateShaders(m_d3d_device, *programs)) return;
			Gfx::CreateOptionalProgram(m_d3d_device, *programs, ProgramId::Resolve, VertShaderId::Passthrough,
				shader_frag_resolve, sizeof(shader_frag_resolve));
			Gfx::CreateOptionalProgram(m_d3d_device, *programs, ProgramId::ResolveRegion, VertShaderId::Passthrough, shader_frag_resolve_region,
				sizeof(shader_frag_resolve_region));
			Gfx::CreateOptionalProgram(m_d3d_device, *programs, ProgramId::ForegroundResolve, VertShaderId::Passthrough,
				shader_frag_foreground_resolve, sizeof(shader_frag_foreground_resolve));
			Gfx::CreateOptionalProgram(m_d3d_device, *programs, ProgramId::BlurPaired, VertShaderId::Blur,
				shader_frag_blur_paired, sizeof(shader_frag_blur_paired));
			program_data = std::move(programs);
			cached_device = m_d3d_device;
			cached_programs = program_data;
		}
	}

	// Create vertex layout. This will be constant to avoid copying to an intermediate struct.
	{
		D3D11_INPUT_ELEMENT_DESC polygonLayout[] = {
			{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
		};
		uint32_t numElements = sizeof(polygonLayout) / sizeof(polygonLayout[0]);

		HRESULT result =
			m_d3d_device->CreateInputLayout(polygonLayout, numElements, program_data->vert_shaders[VertShaderId::Main]->GetBufferPointer(),
				program_data->vert_shaders[VertShaderId::Main]->GetBufferSize(), &m_vertex_layout);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateInputLayout");
		if (FAILED(result))
		{
			return;
		}
	}

	// Create constant buffers. This is so that we can bind uniforms such as translation and color to the shaders.
	{
		D3D11_BUFFER_DESC cbufferDesc{};
		cbufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		cbufferDesc.ByteWidth = sizeof(ShaderCbuffer);
		cbufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		cbufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		cbufferDesc.MiscFlags = 0;
		cbufferDesc.StructureByteStride = 0;

		HRESULT result = m_d3d_device->CreateBuffer(&cbufferDesc, nullptr, &m_shader_buffer);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBuffer");
		if (FAILED(result))
		{
			return;
		}

		cbufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		cbufferDesc.ByteWidth = sizeof(ColorMatrixCbuffer);
		cbufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		cbufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		cbufferDesc.MiscFlags = 0;
		cbufferDesc.StructureByteStride = 0;

		result = m_d3d_device->CreateBuffer(&cbufferDesc, nullptr, &m_color_matrix_cbuffer);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBuffer");
		if (FAILED(result))
		{
			return;
		}

		cbufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		cbufferDesc.ByteWidth = sizeof(ShaderCbuffer::Blur);
		cbufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		cbufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		cbufferDesc.MiscFlags = 0;
		cbufferDesc.StructureByteStride = 0;

		result = m_d3d_device->CreateBuffer(&cbufferDesc, nullptr, &m_blur_cbuffer);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBuffer");
		if (FAILED(result))
		{
			return;
		}

		cbufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		cbufferDesc.ByteWidth = sizeof(ShaderCbuffer::DropShadow);
		cbufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		cbufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		cbufferDesc.MiscFlags = 0;
		cbufferDesc.StructureByteStride = 0;

		result = m_d3d_device->CreateBuffer(&cbufferDesc, nullptr, &m_drop_shadow_cbuffer);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBuffer");
		if (FAILED(result))
		{
			return;
		}
		cbufferDesc.ByteWidth = sizeof(m_sampling_values);
		if (FAILED(m_d3d_device->CreateBuffer(&cbufferDesc, nullptr, &m_sampling_cbuffer)))
			return;
	}

	// Create sampler state for textures
	{
		D3D11_SAMPLER_DESC samplerDesc{};
		samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
		samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
		samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
		samplerDesc.MipLODBias = 0.0f;
		samplerDesc.MaxAnisotropy = 1;
		samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
		samplerDesc.BorderColor[0] = 0;
		samplerDesc.BorderColor[1] = 0;
		samplerDesc.BorderColor[2] = 0;
		samplerDesc.BorderColor[3] = 0;
		samplerDesc.MinLOD = 0;
		samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

		HRESULT result = m_d3d_device->CreateSamplerState(&samplerDesc, &m_sampler_state);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateSamplerState");
		if (FAILED(result))
		{
			return;
		}
		// Framebuffer filtering must not wrap into unrelated pixels at the opposite edge.
		samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
		samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
		samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
		result = m_d3d_device->CreateSamplerState(&samplerDesc, &m_postprocess_sampler_state);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateSamplerState for postprocessing");
		if (FAILED(result))
			return;
	}

	// Create depth stencil states
	{
		D3D11_DEPTH_STENCIL_DESC desc;
		ZeroMemory(&desc, sizeof(desc));
		desc.DepthEnable = false;
		desc.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
		desc.StencilWriteMask = 0;
		desc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
		desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
		desc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
		desc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
		desc.BackFace = desc.FrontFace;
		desc.StencilEnable = true;
		// Disabled
		if (FAILED(m_d3d_device->CreateDepthStencilState(&desc, &m_depth_stencil_state_disable))) return;

		desc.StencilEnable = true;
		desc.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
		desc.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
		desc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
		desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_REPLACE;
		desc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
		desc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
		desc.BackFace = desc.FrontFace;
		// Set and SetInverse
		if (FAILED(m_d3d_device->CreateDepthStencilState(&desc, &m_depth_stencil_state_stencil_set))) return;

		desc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
		desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_INCR;
		desc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
		desc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
		desc.BackFace = desc.FrontFace;
		// Intersect
		if (FAILED(m_d3d_device->CreateDepthStencilState(&desc, &m_depth_stencil_state_stencil_intersect))) return;

		desc.StencilWriteMask = 0;
		desc.FrontFace.StencilFunc = D3D11_COMPARISON_EQUAL;
		desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
		desc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
		desc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
		desc.BackFace = desc.FrontFace;
		// Stencil test
		if (FAILED(m_d3d_device->CreateDepthStencilState(&desc, &m_depth_stencil_state_stencil_test))) return;
	}
	m_valid = true;
}

RenderInterface_DX11::~RenderInterface_DX11()
{
	Dx11CleanupResourceIfCreated(m_dynamic_quad_geometry.vertex_buffer);
	Dx11CleanupResourceIfCreated(m_dynamic_quad_geometry.index_buffer);
	if (m_fullscreen_quad_geometry)
	{
		RenderInterface_DX11::ReleaseGeometry(m_fullscreen_quad_geometry);
		m_fullscreen_quad_geometry = {};
	}

	if (program_data)
	{
		program_data.reset();
	}

	// Cleans up all general resources
	Dx11CleanupResourceIfCreated(m_sampler_state);
	Dx11CleanupResourceIfCreated(m_postprocess_sampler_state);
	Dx11CleanupResourceIfCreated(m_blend_state_enable);
	Dx11CleanupResourceIfCreated(m_blend_state_color_filter);
	Dx11CleanupResourceIfCreated(m_blend_state_disable_color);
	Dx11CleanupResourceIfCreated(m_depth_stencil_state_disable);
	Dx11CleanupResourceIfCreated(m_depth_stencil_state_stencil_intersect);
	Dx11CleanupResourceIfCreated(m_depth_stencil_state_stencil_set);
	Dx11CleanupResourceIfCreated(m_depth_stencil_state_stencil_test);
	Dx11CleanupResourceIfCreated(m_rasterizer_state_scissor_disabled);
	Dx11CleanupResourceIfCreated(m_rasterizer_state_scissor_enabled);
	Dx11CleanupResourceIfCreated(m_shader_buffer);
	Dx11CleanupResourceIfCreated(m_color_matrix_cbuffer);
	Dx11CleanupResourceIfCreated(m_blur_cbuffer);
	Dx11CleanupResourceIfCreated(m_drop_shadow_cbuffer);
	Dx11CleanupResourceIfCreated(m_sampling_cbuffer);
	Dx11CleanupResourceIfCreated(m_vertex_layout);
	Dx11CleanupResourceIfCreated(m_d3d_context);
	Dx11CleanupResourceIfCreated(m_d3d_device_1);
	Dx11CleanupResourceIfCreated(m_d3d_device);
}

void RenderInterface_DX11::BeginFrame(bool preserve_state, const FrameRegions* regions)
{
	RMLUI_ASSERT(m_viewport_width >= 1 && m_viewport_height >= 1);
	RMLUI_ASSERTMSG(m_d3d_context, "d3d_context cannot be nullptr!");
	RMLUI_ASSERTMSG(m_d3d_device, "d3d_device cannot be nullptr!");

	// Backup DX11 state
	m_preserve_state = preserve_state;
	m_frame_regions = regions;
	if (regions && !regions->full && (!program_data->programs[ProgramId::Resolve].pixel_shader))
		m_frame_regions = nullptr;
	if (m_preserve_state)
	{
		m_previous_d3d_state.scissor_rects_count = m_previous_d3d_state.viewports_count = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
		m_d3d_context->RSGetScissorRects(&m_previous_d3d_state.scissor_rects_count, m_previous_d3d_state.scissor_rects);
		m_d3d_context->RSGetViewports(&m_previous_d3d_state.viewports_count, m_previous_d3d_state.viewports);
		m_d3d_context->RSGetState(&m_previous_d3d_state.rasterizer_state);
		m_d3d_context->OMGetBlendState(&m_previous_d3d_state.blend_state, m_previous_d3d_state.blend_factor, &m_previous_d3d_state.sample_mask);
		m_d3d_context->OMGetDepthStencilState(&m_previous_d3d_state.depth_stencil_state, &m_previous_d3d_state.stencil_ref);
		m_d3d_context->PSGetShaderResources(0, 1, &m_previous_d3d_state.pixel_shader_shader_resource);
		m_d3d_context->PSGetSamplers(0, 1, &m_previous_d3d_state.pixel_shader_sampler);
		m_previous_d3d_state.pixel_shader_instances_count = m_previous_d3d_state.vertex_shader_instances_count =
			m_previous_d3d_state.geometry_shader_instances_count = 256;
		m_d3d_context->PSGetShader(&m_previous_d3d_state.pixel_shader, m_previous_d3d_state.pixel_shader_instances,
			&m_previous_d3d_state.pixel_shader_instances_count);
		m_d3d_context->VSGetShader(&m_previous_d3d_state.vertex_shader, m_previous_d3d_state.vertex_shader_instances,
			&m_previous_d3d_state.vertex_shader_instances_count);
		m_d3d_context->VSGetConstantBuffers(0, 1, &m_previous_d3d_state.vertex_shader_constant_buffer);
		m_d3d_context->VSGetConstantBuffers(2, 1, &m_previous_d3d_state.vertex_sampling_buffer);
		m_d3d_context->PSGetConstantBuffers(2, 1, &m_previous_d3d_state.pixel_sampling_buffer);
		m_d3d_context->GSGetShader(&m_previous_d3d_state.geometry_shader, m_previous_d3d_state.geometry_shader_instances,
			&m_previous_d3d_state.geometry_shader_instances_count);

		m_d3d_context->IAGetPrimitiveTopology(&m_previous_d3d_state.primitive_topology);
		m_d3d_context->IAGetIndexBuffer(&m_previous_d3d_state.index_buffer, &m_previous_d3d_state.index_buffer_format,
			&m_previous_d3d_state.index_buffer_offset);
		m_d3d_context->IAGetVertexBuffers(0, 1, &m_previous_d3d_state.vertex_buffer, &m_previous_d3d_state.vertex_buffer_stride,
			&m_previous_d3d_state.vertex_buffer_offset);
		m_d3d_context->IAGetInputLayout(&m_previous_d3d_state.input_layout);
	}

	// Initialise DX11 state for RmlUi
	D3D11_VIEWPORT d3dviewport;
	d3dviewport.TopLeftX = 0;
	d3dviewport.TopLeftY = 0;
	d3dviewport.Width = (float)m_viewport_width;
	d3dviewport.Height = (float)m_viewport_height;
	d3dviewport.MinDepth = 0.0f;
	d3dviewport.MaxDepth = 1.0f;
	m_d3d_context->RSSetViewports(1, &d3dviewport);

	SetBlendState(m_blend_state_enable);
	m_d3d_context->RSSetState(m_rasterizer_state_scissor_disabled); // Disable scissor
	m_stencil_ref_value = 0;                                        // Reset stencil
	m_d3d_context->OMSetDepthStencilState(m_depth_stencil_state_disable, 0);

	SetTransform(nullptr);

	m_clip_commands.clear();
	++m_clip_generation;
	m_is_stencil_enabled = false;
	m_scissor_state = Rml::Rectanglei::MakeInvalid();
	m_scissor_binding_dirty = true;
	m_render_layers->BeginFrame(m_canvas_width, m_canvas_height);
	BindTarget(m_render_layers->GetTopLayer());
	float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	if (!m_frame_regions || !m_frame_regions->opaque_background)
	{
		if (m_frame_regions && !m_frame_regions->full)
		{
			std::vector<D3D11_RECT> rectangles;
			for (const auto& r : m_frame_regions->clear) rectangles.push_back({r.Left(), r.Top(), r.Right(), r.Bottom()});
			if (!rectangles.empty()) m_d3d_context->ClearView(m_render_layers->GetTopLayer().render_target_view,
				clearColor, rectangles.data(), UINT(rectangles.size()));
		}
		else m_d3d_context->ClearRenderTargetView(m_render_layers->GetTopLayer().render_target_view, clearColor);
	}

	UseProgram(ProgramId::None);
	m_scissor_state = Rml::Rectanglei::MakeInvalid();
}

void RenderInterface_DX11::EndFrame(ID3D11RenderTargetView* p_render_target_view)
{
	m_render_layers->SelectPostprocess(0);
	RMLUI_ASSERTMSG(p_render_target_view, "p_render_target_view cannot be nullptr!");
	if (m_frame_regions && !m_frame_regions->full)
	{
		if (m_frame_regions->output.empty()) { FinishFrame(); return; }
		const auto& source = m_render_layers->GetTopLayer();
		const auto& destination = m_render_layers->GetResolveTarget(m_render_layers->GetTopLayer());
		ID3D11ShaderResourceView* null_views[2] = {};
		// Resolve each non-overlapping output rectangle once; stale pixels outside it are never sampled.
		m_d3d_context->PSSetShaderResources(0, 2, null_views);
		BindTarget(destination);
		DisableBlend();
		UseProgram(ProgramId::Resolve);
		// RenderGeometry unbinds t0 after each draw, including fullscreen quads.
		for (const auto& region : m_frame_regions->output)
		{
			BindTexture(source);
			SetScissor(region);
			DrawFullscreenQuad();
		}
		m_d3d_context->PSSetShaderResources(0, 2, null_views);
		m_d3d_context->OMSetRenderTargets(1, &p_render_target_view, nullptr);
		m_d3d_context->PSSetSamplers(0, 1, &m_postprocess_sampler_state);
		SetBlendState(m_blend_state_enable);
		UseProgram(ProgramId::Passthrough);
		for (const auto& region : m_frame_regions->output)
		{
			BindTexture(destination);
			SetScissor(region);
			DrawFullscreenQuad();
		}
		FinishFrame();
		return;
	}

	const Gfx::RenderTargetData& rt_active = m_render_layers->GetTopLayer();
	const Gfx::RenderTargetData& rt_postprocess = m_render_layers->GetResolveTarget(m_render_layers->GetTopLayer());

	// Resolve MSAA to the post-process framebuffer.
	m_d3d_context->ResolveSubresource(rt_postprocess.render_target_texture, 0, rt_active.render_target_texture, 0, DXGI_FORMAT_R8G8B8A8_UNORM);

	// Draw to bound_render_target (usually the swapchain RTV)
	m_d3d_context->OMSetRenderTargets(1, &p_render_target_view, nullptr);

	// Assuming we have an opaque background, we can just write to it with the premultiplied alpha blend mode and we'll get the correct result.
	// Instead, if we had a transparent destination that didn't use premultiplied alpha, we would need to perform a manual un-premultiplication step.
	BindTexture(rt_postprocess);
	m_d3d_context->PSSetSamplers(0, 1, &m_postprocess_sampler_state);

	UseProgram(ProgramId::Passthrough);

	DrawFullscreenQuad();

	FinishFrame();
}

void RenderInterface_DX11::FinishFrame()
{
	m_frame_regions = nullptr;
	m_render_layers->EndFrame();
	m_clip_commands.clear();

	// Reset internal state
	m_current_blend_state = nullptr;

	// Restore modified DX state
	if (m_preserve_state)
	{
		m_d3d_context->RSSetScissorRects(m_previous_d3d_state.scissor_rects_count, m_previous_d3d_state.scissor_rects);
		m_d3d_context->RSSetViewports(m_previous_d3d_state.viewports_count, m_previous_d3d_state.viewports);
		m_d3d_context->RSSetState(m_previous_d3d_state.rasterizer_state);
		if (m_previous_d3d_state.rasterizer_state)
		{
			m_previous_d3d_state.rasterizer_state->Release();
		}
		m_d3d_context->OMSetBlendState(m_previous_d3d_state.blend_state, m_previous_d3d_state.blend_factor, m_previous_d3d_state.sample_mask);
		if (m_previous_d3d_state.blend_state)
		{
			m_previous_d3d_state.blend_state->Release();
		}
		m_d3d_context->OMSetDepthStencilState(m_previous_d3d_state.depth_stencil_state, m_previous_d3d_state.stencil_ref);
		if (m_previous_d3d_state.depth_stencil_state)
		{
			m_previous_d3d_state.depth_stencil_state->Release();
		}
		m_d3d_context->PSSetShaderResources(0, 1, &m_previous_d3d_state.pixel_shader_shader_resource);
		if (m_previous_d3d_state.pixel_shader_shader_resource)
		{
			m_previous_d3d_state.pixel_shader_shader_resource->Release();
		}
		m_d3d_context->PSSetSamplers(0, 1, &m_previous_d3d_state.pixel_shader_sampler);
		if (m_previous_d3d_state.pixel_shader_sampler)
		{
			m_previous_d3d_state.pixel_shader_sampler->Release();
		}
		m_d3d_context->PSSetShader(m_previous_d3d_state.pixel_shader, m_previous_d3d_state.pixel_shader_instances,
			m_previous_d3d_state.pixel_shader_instances_count);
		if (m_previous_d3d_state.pixel_shader)
		{
			m_previous_d3d_state.pixel_shader->Release();
		}
		for (UINT i = 0; i < m_previous_d3d_state.pixel_shader_instances_count; i++)
		{
			if (m_previous_d3d_state.pixel_shader_instances[i])
			{
				m_previous_d3d_state.pixel_shader_instances[i]->Release();
			}
		}
		m_d3d_context->VSSetShader(m_previous_d3d_state.vertex_shader, m_previous_d3d_state.vertex_shader_instances,
			m_previous_d3d_state.vertex_shader_instances_count);
		if (m_previous_d3d_state.vertex_shader)
		{
			m_previous_d3d_state.vertex_shader->Release();
		}
		m_d3d_context->VSSetConstantBuffers(0, 1, &m_previous_d3d_state.vertex_shader_constant_buffer);
		m_d3d_context->VSSetConstantBuffers(2, 1, &m_previous_d3d_state.vertex_sampling_buffer);
		m_d3d_context->PSSetConstantBuffers(2, 1, &m_previous_d3d_state.pixel_sampling_buffer);
		Dx11CleanupResourceIfCreated(m_previous_d3d_state.vertex_sampling_buffer);
		Dx11CleanupResourceIfCreated(m_previous_d3d_state.pixel_sampling_buffer);
		if (m_previous_d3d_state.vertex_shader_constant_buffer)
		{
			m_previous_d3d_state.vertex_shader_constant_buffer->Release();
		}
		m_d3d_context->GSSetShader(m_previous_d3d_state.geometry_shader, m_previous_d3d_state.geometry_shader_instances,
			m_previous_d3d_state.geometry_shader_instances_count);
		if (m_previous_d3d_state.geometry_shader)
		{
			m_previous_d3d_state.geometry_shader->Release();
		}
		for (UINT i = 0; i < m_previous_d3d_state.vertex_shader_instances_count; i++)
		{
			if (m_previous_d3d_state.vertex_shader_instances[i])
			{
				m_previous_d3d_state.vertex_shader_instances[i]->Release();
			}
		}
		m_d3d_context->IASetPrimitiveTopology(m_previous_d3d_state.primitive_topology);
		m_d3d_context->IASetIndexBuffer(m_previous_d3d_state.index_buffer, m_previous_d3d_state.index_buffer_format,
			m_previous_d3d_state.index_buffer_offset);
		if (m_previous_d3d_state.index_buffer)
		{
			m_previous_d3d_state.index_buffer->Release();
		}
		m_d3d_context->IASetVertexBuffers(0, 1, &m_previous_d3d_state.vertex_buffer, &m_previous_d3d_state.vertex_buffer_stride,
			&m_previous_d3d_state.vertex_buffer_offset);
		if (m_previous_d3d_state.vertex_buffer)
		{
			m_previous_d3d_state.vertex_buffer->Release();
		}
		m_d3d_context->IASetInputLayout(m_previous_d3d_state.input_layout);
		if (m_previous_d3d_state.input_layout)
		{
			m_previous_d3d_state.input_layout->Release();
		}
	}
}

Rml::CompiledGeometryHandle RenderInterface_DX11::CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices)
{
	ID3D11Buffer* vertex_buffer = nullptr;
	ID3D11Buffer* index_buffer = nullptr;

	// Bind vertex buffer
	{
		D3D11_BUFFER_DESC vertexBufferDesc{};
		vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
		vertexBufferDesc.ByteWidth = static_cast<UINT>(sizeof(Rml::Vertex) * vertices.size());
		vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		vertexBufferDesc.CPUAccessFlags = 0;
		vertexBufferDesc.MiscFlags = 0;
		vertexBufferDesc.StructureByteStride = 0;

		D3D11_SUBRESOURCE_DATA vertexData{};
		vertexData.pSysMem = vertices.data();
		vertexData.SysMemPitch = 0;
		vertexData.SysMemSlicePitch = 0;

		HRESULT result = m_d3d_device->CreateBuffer(&vertexBufferDesc, &vertexData, &vertex_buffer);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBuffer");
		if (FAILED(result))
		{
			return Rml::CompiledGeometryHandle(0);
		}
	}

	// Bind index buffer
	{
		D3D11_BUFFER_DESC indexBufferDesc{};
		indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
		indexBufferDesc.ByteWidth = static_cast<UINT>(sizeof(int) * indices.size());
		indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		indexBufferDesc.CPUAccessFlags = 0;
		indexBufferDesc.MiscFlags = 0;
		indexBufferDesc.StructureByteStride = 0;

		D3D11_SUBRESOURCE_DATA indexData{};
		indexData.pSysMem = indices.data();
		indexData.SysMemPitch = 0;
		indexData.SysMemSlicePitch = 0;

		HRESULT result = m_d3d_device->CreateBuffer(&indexBufferDesc, &indexData, &index_buffer);
		RMLUI_DX_ASSERTMSG(result, "failed to CreateBuffer");
		if (FAILED(result))
		{
			Dx11CleanupResourceIfCreated(vertex_buffer);
			return Rml::CompiledGeometryHandle(0);
		}
	}

	DX11_GeometryData* geometryData = new DX11_GeometryData;

	geometryData->vertex_buffer = vertex_buffer;
	geometryData->index_buffer = index_buffer;
	geometryData->index_count = indices.size();

	return (Rml::CompiledGeometryHandle)geometryData;
}

void RenderInterface_DX11::ReleaseGeometry(Rml::CompiledGeometryHandle handle)
{
	DX11_GeometryData* geometry = (DX11_GeometryData*)handle;

	Dx11CleanupResourceIfCreated(geometry->vertex_buffer);
	Dx11CleanupResourceIfCreated(geometry->index_buffer);

	delete geometry;
}

void RenderInterface_DX11::RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture)
{
	DX11_GeometryData* geometry = (DX11_GeometryData*)handle;
	if (!m_drawing_clip && texture != TexturePostprocess && texture != TexturePostprocessNoBinding)
		EnsureClip(m_render_layers->GetTopLayer());

	if (m_translation != translation)
	{
		m_translation = translation;
		m_cbuffer_dirty = true;
	}

	if (texture == TexturePostprocess || texture == TexturePostprocessNoBinding)
	{
		// Do nothing.
	}
	else if (texture)
	{
		// Texture available
		UseProgram(ProgramId::Texture);
		if (texture != TextureEnableWithoutBinding)
		{
			ID3D11ShaderResourceView* texture_view = (ID3D11ShaderResourceView*)texture;
			m_d3d_context->PSSetShaderResources(0, 1, &texture_view);
		}
		m_d3d_context->PSSetSamplers(0, 1, &m_sampler_state);
	}
	else
	{
		// No texture, use color
		UseProgram(ProgramId::Color);
	}

	if (texture != TexturePostprocessNoBinding)
	{
		UpdateConstantBuffer();

		m_d3d_context->VSSetConstantBuffers(0, 1, &m_shader_buffer);
		m_d3d_context->PSSetConstantBuffers(0, 1, &m_shader_buffer);
	}

	// Bind vertex and index buffers, issue draw call
	uint32_t stride = sizeof(Rml::Vertex);
	uint32_t offset = 0;
	m_d3d_context->IASetInputLayout(m_vertex_layout);
	m_d3d_context->IASetVertexBuffers(0, 1, &geometry->vertex_buffer, &stride, &offset);
	m_d3d_context->IASetIndexBuffer(geometry->index_buffer, DXGI_FORMAT_R32_UINT, 0);
	m_d3d_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	m_d3d_context->DrawIndexed(static_cast<UINT>(geometry->index_count), 0, 0);

	// Unbind resources
	ID3D11ShaderResourceView* const nullSRV = nullptr;
	m_d3d_context->PSSetShaderResources(0, 1, &nullSRV);
}

// Set to byte packing, or the compiler will expand our struct, which means it won't read correctly from file
#pragma pack(1)
struct TGAHeader {
	char idLength;
	char colourMapType;
	char dataType;
	short int colourMapOrigin;
	short int colourMapLength;
	char colourMapDepth;
	short int xOrigin;
	short int yOrigin;
	short int width;
	short int height;
	char bitsPerPixel;
	char imageDescriptor;
};
// Restore packing
#pragma pack()

Rml::TextureHandle RenderInterface_DX11::LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source)
{
	Rml::FileInterface* file_interface = Rml::GetFileInterface();
	Rml::FileHandle file_handle = file_interface->Open(source);
	if (!file_handle)
	{
		return false;
	}

	file_interface->Seek(file_handle, 0, SEEK_END);
	size_t buffer_size = file_interface->Tell(file_handle);
	file_interface->Seek(file_handle, 0, SEEK_SET);

	if (buffer_size <= sizeof(TGAHeader))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Texture file size is smaller than TGAHeader, file is not a valid TGA image.");
		file_interface->Close(file_handle);
		return false;
	}

	using Rml::byte;
	Rml::UniquePtr<byte[]> buffer(new byte[buffer_size]);
	file_interface->Read(buffer.get(), buffer_size, file_handle);
	file_interface->Close(file_handle);

	TGAHeader header;
	memcpy(&header, buffer.get(), sizeof(TGAHeader));

	int color_mode = header.bitsPerPixel / 8;
	const size_t image_size = header.width * header.height * 4; // We always make 32bit textures

	if (header.dataType != 2)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Only 24/32bit uncompressed TGAs are supported.");
		return false;
	}

	// Ensure we have at least 3 colors
	if (color_mode < 3)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Only 24 and 32bit textures are supported.");
		return false;
	}

	const byte* image_src = buffer.get() + sizeof(TGAHeader);
	Rml::UniquePtr<byte[]> image_dest_buffer(new byte[image_size]);
	byte* image_dest = image_dest_buffer.get();

	// Targa is BGR, swap to RGB, flip Y axis, and convert to premultiplied alpha.
	for (long y = 0; y < header.height; y++)
	{
		long read_index = y * header.width * color_mode;
		long write_index = ((header.imageDescriptor & 32) != 0) ? read_index : (header.height - y - 1) * header.width * 4;
		for (long x = 0; x < header.width; x++)
		{
			image_dest[write_index] = image_src[read_index + 2];
			image_dest[write_index + 1] = image_src[read_index + 1];
			image_dest[write_index + 2] = image_src[read_index];
			if (color_mode == 4)
			{
				const byte alpha = image_src[read_index + 3];
				for (size_t j = 0; j < 3; j++)
					image_dest[write_index + j] = byte((image_dest[write_index + j] * alpha) / 255);
				image_dest[write_index + 3] = alpha;
			}
			else
				image_dest[write_index + 3] = 255;

			write_index += 4;
			read_index += color_mode;
		}
	}

	texture_dimensions.x = header.width;
	texture_dimensions.y = header.height;

	return GenerateTexture({image_dest, image_size}, texture_dimensions);
}

Rml::TextureHandle RenderInterface_DX11::GenerateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions)
{
	ID3D11Texture2D* gpu_texture = nullptr;
	ID3D11ShaderResourceView* gpu_texture_view = nullptr;

	D3D11_TEXTURE2D_DESC textureDesc{};
	textureDesc.Width = source_dimensions.x;
	textureDesc.Height = source_dimensions.y;
	textureDesc.MipLevels = 0;
	textureDesc.ArraySize = 1;
	textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.SampleDesc.Quality = 0;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

	HRESULT result = m_d3d_device->CreateTexture2D(&textureDesc, nullptr, &gpu_texture);
	RMLUI_DX_ASSERTMSG(result, "failed to CreateTexture2D");
	if (FAILED(result))
	{
		return Rml::TextureHandle(0);
	}

	// Set the row pitch of the raw image data
	uint32_t rowPitch = (source_dimensions.x * 4) * sizeof(unsigned char);

	// Copy the raw image data into the texture
	if (source.size() > 0)
	{
		m_d3d_context->UpdateSubresource(gpu_texture, 0, nullptr, source.data(), rowPitch, 0);
	}

	// Setup the shader resource view description.
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
	srvDesc.Format = textureDesc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = static_cast<UINT>(-1);

	// Create the shader resource view for the texture.
	result = m_d3d_device->CreateShaderResourceView(gpu_texture, &srvDesc, &gpu_texture_view);
	RMLUI_DX_ASSERTMSG(result, "failed to CreateShaderResourceView");
	if (FAILED(result))
	{
		Dx11CleanupResourceIfCreated(gpu_texture);
		return Rml::TextureHandle(0);
	}
	Dx11CleanupResourceIfCreated(gpu_texture);

	// Generate mipmaps for this texture.
	if (source.size() > 0)
	{
		m_d3d_context->GenerateMips(gpu_texture_view);
	}

	return Rml::TextureHandle(gpu_texture_view);
}

void RenderInterface_DX11::ReleaseTexture(Rml::TextureHandle texture_handle)
{
	ID3D11ShaderResourceView* texture_view = (ID3D11ShaderResourceView*)texture_handle;
	Dx11CleanupResourceIfCreated(texture_view);
}

static Rml::Colourf ConvertToColorf(Rml::ColourbPremultiplied c0)
{
	Rml::Colourf result;
	for (int i = 0; i < 4; i++)
		result[i] = (1.f / 255.f) * float(c0[i]);
	return result;
}

void RenderInterface_DX11::DrawFullscreenQuad()
{
	RenderGeometry(m_fullscreen_quad_geometry, {}, RenderInterface_DX11::TexturePostprocess);
}

void RenderInterface_DX11::DrawQuad(Rml::Span<const Rml::Vertex> vertices)
{
	RMLUI_ASSERT(vertices.size() == 4);
	auto draw_transient = [&]()
	{
		const int indices[] = {0, 3, 1, 1, 3, 2};
		const auto geometry = CompileGeometry(vertices, {indices, 6});
		if (geometry)
		{
			RenderGeometry(geometry, {}, TexturePostprocess);
			ReleaseGeometry(geometry);
		}
	};
	if (!m_dynamic_quad_geometry.vertex_buffer)
	{
		draw_transient();
		return;
	}
	D3D11_MAPPED_SUBRESOURCE mapped{};
	const HRESULT result = m_d3d_context->Map(m_dynamic_quad_geometry.vertex_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
	if (FAILED(result))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to update the dynamic quad buffer (%d)", result);
		draw_transient();
		return;
	}
	// DISCARD gives queued draws their own backing storage; one quad cannot overwrite an earlier draw.
	memcpy(mapped.pData, vertices.data(), 4 * sizeof(Rml::Vertex));
	m_d3d_context->Unmap(m_dynamic_quad_geometry.vertex_buffer, 0);
	RenderGeometry(reinterpret_cast<Rml::CompiledGeometryHandle>(&m_dynamic_quad_geometry), {}, TexturePostprocess);
}

void RenderInterface_DX11::DrawFullscreenQuad(Rml::Vector2f uv_offset, Rml::Vector2f uv_scaling)
{
	if (uv_offset == Rml::Vector2f() && uv_scaling == Rml::Vector2f(1.f))
	{
		DrawFullscreenQuad();
		return;
	}
	Rml::Vertex vertices[4] = {
		{{-1.f, -1.f}, {}, {0.f, 0.f}}, {{1.f, -1.f}, {}, {1.f, 0.f}},
		{{1.f, 1.f}, {}, {1.f, 1.f}}, {{-1.f, 1.f}, {}, {0.f, 1.f}}
	};
	for (Rml::Vertex& vertex : vertices)
		vertex.tex_coord = vertex.tex_coord * uv_scaling + uv_offset;
	DrawQuad({vertices, 4});
}

/// Flip vertical axis of the rectangle, and move its origin to the vertically opposite side of the viewport.
/// @note Changes coordinate system from RmlUi to OpenGL, or equivalently in reverse.
/// @note The Rectangle::Top and Rectangle::Bottom members will have reverse meaning in the returned rectangle.
static Rml::Rectanglei VerticallyFlipped(Rml::Rectanglei rect, int viewport_height)
{
	RMLUI_ASSERT(rect.Valid());
	Rml::Rectanglei flipped_rect = rect;
	flipped_rect.p0.y = viewport_height - rect.p1.y;
	flipped_rect.p1.y = viewport_height - rect.p0.y;
	return flipped_rect;
}

Rml::Rectanglei RenderInterface_DX11::LocalBounds(Rml::Rectanglei bounds) const
{
	if (bounds.Valid())
	{
		bounds.p0 -= m_target_origin;
		bounds.p1 -= m_target_origin;
	}
	return bounds;
}

void RenderInterface_DX11::SetLocalScissor(Rml::Rectanglei region, bool vertically_flip)
{
	if (region.Valid())
	{
		region.p0 += m_target_origin;
		region.p1 += m_target_origin;
	}
	SetScissor(region, vertically_flip);
}

void RenderInterface_DX11::SetScissor(Rml::Rectanglei region, bool vertically_flip)
{
	if (region.Valid() && vertically_flip)
	{
		auto local = VerticallyFlipped(LocalBounds(region), m_viewport_height);
		region = Rml::Rectanglei::FromPositionSize(local.p0 + m_target_origin, local.Size());
	}
	const bool canvas_clip = m_target_is_layer &&
		(m_target_origin.x < 0 || m_target_origin.y < 0 || m_target_origin.x + m_viewport_width > m_canvas_width ||
			m_target_origin.y + m_viewport_height > m_canvas_height);
	const bool enabled = region.Valid() || canvas_clip;
	if (enabled != m_scissor_enabled || m_scissor_binding_dirty)
	{
		m_d3d_context->RSSetState(enabled ? m_rasterizer_state_scissor_enabled : m_rasterizer_state_scissor_disabled);
		m_scissor_enabled = enabled;
	}
	if (enabled && (region != m_scissor_state || m_scissor_binding_dirty))
	{
		const auto canvas = Rml::Rectanglei::FromSize({m_canvas_width, m_canvas_height});
		const auto clipped = canvas_clip ? (region.Valid() ? region.Intersect(canvas) : canvas) : region;
		const auto local = LocalBounds(clipped);
		D3D11_RECT scissor{Rml::Math::Clamp(local.Left(), 0, m_viewport_width), Rml::Math::Clamp(local.Top(), 0, m_viewport_height),
			Rml::Math::Clamp(local.Right(), 0, m_viewport_width), Rml::Math::Clamp(local.Bottom(), 0, m_viewport_height)};
		m_d3d_context->RSSetScissorRects(1, &scissor);
	}
	m_scissor_state = region;
	m_scissor_binding_dirty = false;
}

void RenderInterface_DX11::EnableScissorRegion(bool enable)
{
	// Assume enable is immediately followed by a SetScissorRegion() call, and ignore it here.
	if (!enable)
		SetScissor(Rml::Rectanglei::MakeInvalid(), false);
}

void RenderInterface_DX11::SetScissorRegion(Rml::Rectanglei region)
{
	SetScissor(RasterBounds(region));
}

void RenderInterface_DX11::SetViewport(const int width, const int height)
{
	if (!m_surface_target && (m_canvas_width != width || m_canvas_height != height))
	{
		m_default_render_layers.Reset();
		m_mask_pool.clear();
	}
	m_canvas_width = m_viewport_width = width;
	m_canvas_height = m_viewport_height = height;
	m_target_origin = {};
	m_raster_scale = {1, 1};
	m_projection = Rml::Matrix4f::ProjectOrtho(0, (float)width, (float)height, 0, -10000, 10000);
}

void RenderInterface_DX11::SetSurfaceViewport(int logical_width, int logical_height, int raster_width, int raster_height)
{
	SetViewport(raster_width, raster_height);
	m_raster_scale = {float(raster_width) / logical_width, float(raster_height) / logical_height};
	m_projection = Rml::Matrix4f::ProjectOrtho(0, float(logical_width), float(logical_height), 0, -10000, 10000);
}

Rml::Rectanglei RenderInterface_DX11::RasterBounds(Rml::Rectanglei bounds) const
{
	if (!bounds.Valid() || m_raster_scale == Rml::Vector2f{1, 1}) return bounds;
	const auto lower = [](float value) { return int(std::floor(Rml::Math::Clamp(value, -32768.f, 32767.f))); };
	const auto upper = [](float value) { return int(std::ceil(Rml::Math::Clamp(value, -32768.f, 32767.f))); };
	return Rml::Rectanglei::FromCorners({lower(bounds.Left() * m_raster_scale.x), lower(bounds.Top() * m_raster_scale.y)},
		{upper(bounds.Right() * m_raster_scale.x), upper(bounds.Bottom() * m_raster_scale.y)});
}

void RenderInterface_DX11::Clear(ID3D11RenderTargetView* p_render_target_view)
{
	float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	m_d3d_context->ClearRenderTargetView(p_render_target_view, clearColor);
}

void RenderInterface_DX11::RestorePartialRenderState()
{
	UseProgram(ProgramId::None);

	ID3D11BlendState* restore_blend_state = m_current_blend_state;
	m_current_blend_state = nullptr;
	SetBlendState(restore_blend_state);
}

void RenderInterface_DX11::SetBlendState(ID3D11BlendState* blend_state)
{
	if (!m_current_blend_state_enabled || blend_state != m_current_blend_state)
	{
		const float blend_factor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		m_d3d_context->OMSetBlendState(blend_state, blend_factor, 0xFFFFFFFF);
		m_current_blend_state = blend_state;
		m_current_blend_state_enabled = true;
	}
}

void RenderInterface_DX11::DisableBlend()
{
	if (m_current_blend_state_enabled)
	{
		m_d3d_context->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
		m_current_blend_state_enabled = false;
	}
}

void RenderInterface_DX11::SetTransform(const Rml::Matrix4f* new_transform)
{
	m_user_transform = new_transform ? *new_transform : Rml::Matrix4f::Identity();
	m_transform = m_projection * m_user_transform;
	m_cbuffer_dirty = true;
}

void RenderInterface_DX11::SetTargetSpace(int width, int height, Rml::Vector2i origin, bool is_layer)
{
	m_target_is_layer = is_layer;
	m_viewport_width = width;
	m_viewport_height = height;
	m_target_origin = origin;
	D3D11_VIEWPORT viewport{};
	viewport.Width = float(width);
	viewport.Height = float(height);
	viewport.MaxDepth = 1;
	m_d3d_context->RSSetViewports(1, &viewport);
	m_projection = Rml::Matrix4f::ProjectOrtho(origin.x / m_raster_scale.x, (origin.x + width) / m_raster_scale.x,
		(origin.y + height) / m_raster_scale.y, origin.y / m_raster_scale.y, -10000, 10000);
	m_transform = m_projection * m_user_transform;
	m_cbuffer_dirty = true;
	m_scissor_binding_dirty = true;
	SetScissor(m_scissor_state);
}

void RenderInterface_DX11::BindTarget(const Gfx::RenderTargetData& target, bool update_space)
{
	Gfx::BindRenderTarget(m_d3d_context, target);
	if (update_space)
		SetTargetSpace(Rml::Math::Max(1, target.bounds.Width()), Rml::Math::Max(1, target.bounds.Height()), target.bounds.p0,
			target.depth_stencil_view != nullptr);
}

void RenderInterface_DX11::BindTexture(const Gfx::RenderTargetData& texture, uint32_t slot, bool logical_uv)
{
	Gfx::BindTexture(m_d3d_context, texture, slot);
	if (slot)
		return;
	const float width = float(Rml::Math::Max(1, texture.bounds.Width())), height = float(Rml::Math::Max(1, texture.bounds.Height()));
	const float values[8] = {logical_uv ? width / texture.width : 1.f, logical_uv ? height / texture.height : 1.f, 0, 0, .5f / texture.width,
		.5f / texture.height, (width - .5f) / texture.width, (height - .5f) / texture.height};
	if (!m_sampling_initialized || memcmp(values, m_sampling_values, sizeof(values)))
	{
		D3D11_MAPPED_SUBRESOURCE mapping{};
		if (SUCCEEDED(m_d3d_context->Map(m_sampling_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapping)))
		{
			memcpy(mapping.pData, values, sizeof(values));
			m_d3d_context->Unmap(m_sampling_cbuffer, 0);
			memcpy(m_sampling_values, values, sizeof(values));
			m_sampling_initialized = true;
		}
	}
	m_d3d_context->VSSetConstantBuffers(2, 1, &m_sampling_cbuffer);
	m_d3d_context->PSSetConstantBuffers(2, 1, &m_sampling_cbuffer);
}

void RenderInterface_DX11::DrawTargetQuad(const Gfx::RenderTargetData& source, const Gfx::RenderTargetData& destination)
{
	if (source.bounds.Width() == destination.bounds.Width() && source.bounds.Height() == destination.bounds.Height() &&
		source.bounds.p0 == destination.bounds.p0)
	{
		DrawFullscreenQuad();
		return;
	}
	const auto offset = source.bounds.p0 - destination.bounds.p0;
	const float x0 = 2.f * offset.x / destination.bounds.Width() - 1.f,
				x1 = 2.f * (offset.x + source.bounds.Width()) / destination.bounds.Width() - 1.f;
	const float y1 = 1.f - 2.f * offset.y / destination.bounds.Height(),
				y0 = 1.f - 2.f * (offset.y + source.bounds.Height()) / destination.bounds.Height();
	const Rml::Vertex vertices[4] = {{{x0, y0}, {}, {0, 0}}, {{x1, y0}, {}, {1, 0}}, {{x1, y1}, {}, {1, 1}}, {{x0, y1}, {}, {0, 1}}};
	DrawQuad({vertices, 4});
}

void RenderInterface_DX11::BlitRenderTarget(const Gfx::RenderTargetData& source, const Gfx::RenderTargetData& dest, int srcX0, int srcY0, int srcX1,
	int srcY1, int dstX0, int dstY0, int dstX1, int dstY1)
{
	const int dest_width = dstX1 - dstX0;
	const int dest_height = dstY1 - dstY0;
	const bool full_copy = srcX0 == 0 && srcY0 == 0 && dstX0 == 0 && dstY0 == 0
		&& srcX1 == source.width && srcY1 == source.height && dstX1 == dest.width && dstY1 == dest.height
		&& source.width == dest.width && source.height == dest.height && dest.sample_count == 1;

	auto copy_or_resolve = [&](const Gfx::RenderTargetData& target)
	{
		if (source.sample_count > 1)
		{
			m_d3d_context->ResolveSubresource(target.render_target_texture, 0, source.render_target_texture, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
		}
		else
		{
			m_d3d_context->CopyResource(target.render_target_texture, source.render_target_texture);
		}
	};

	ID3D11ShaderResourceView* null_views[2] = {nullptr, nullptr};
	if (full_copy)
	{
		if (source.render_target_texture != dest.render_target_texture)
		{
			m_d3d_context->PSSetShaderResources(0, 2, null_views);
			copy_or_resolve(dest);
		}
		return;
	}

	const auto original_origin = m_target_origin;
	const bool original_is_layer = m_target_is_layer;
	const int original_width = m_viewport_width, original_height = m_viewport_height;
	ID3D11RenderTargetView* original_target = nullptr;
	ID3D11DepthStencilView* original_depth = nullptr;
	m_d3d_context->OMGetRenderTargets(1, &original_target, &original_depth);
	ID3D11BlendState* original_blend = m_current_blend_state;
	const bool original_blend_enabled = m_current_blend_state_enabled;
	m_d3d_context->PSSetShaderResources(0, 2, null_views);
	DisableBlend();

	const Gfx::RenderTargetData* sampled = &source;
	if (source.sample_count > 1 || source.render_target_texture == dest.render_target_texture)
	{
		const auto* scratch = &m_render_layers->GetResolveTarget(source);
		auto aliases = [&](const Gfx::RenderTargetData* candidate)
		{
			return candidate->render_target_texture == source.render_target_texture || candidate->render_target_texture == dest.render_target_texture;
		};
		if (aliases(scratch)) scratch = &m_render_layers->GetPostprocessTertiary();
		if (aliases(scratch)) scratch = &m_render_layers->GetPostprocessSecondary();
		RMLUI_ASSERT(!aliases(scratch));
		m_d3d_context->OMSetRenderTargets(0, nullptr, nullptr);
		copy_or_resolve(*scratch);
		sampled = scratch;
	}

	// Rebinding the output releases source from its RTV slot before it becomes an SRV.
	BindTarget(dest);
	D3D11_VIEWPORT viewport{};
	viewport.Width = float(dest.bounds.Width());
	viewport.Height = float(dest.bounds.Height());
	viewport.MaxDepth = 1.f;
	m_d3d_context->RSSetViewports(1, &viewport);
	m_d3d_context->PSSetSamplers(0, 1, &m_postprocess_sampler_state);
	BindTexture(*sampled, 0, false);
	UseProgram(ProgramId::Passthrough);

	const float u0 = float(srcX0) / float(source.width);
	const int padding_y = source.height - source.bounds.Height();
	const float v0 = float(padding_y + srcY1) / source.height;
	const float u1 = float(srcX1) / float(source.width);
	const float v1 = float(padding_y + srcY0) / source.height;
	const float x0 = (dstX0 / float(dest.bounds.Width())) * 2.f - 1.f;
	const float y0 = ((dest.bounds.Height() - dstY0 - dest_height) / float(dest.bounds.Height())) * 2.f - 1.f;
	const float x1 = ((dstX0 + dest_width) / float(dest.bounds.Width())) * 2.f - 1.f;
	const float y1 = ((dest.bounds.Height() - dstY0) / float(dest.bounds.Height())) * 2.f - 1.f;
	const Rml::Vertex vertices[4] = {
		{{x0, y0}, {}, {u0, v0}}, {{x1, y0}, {}, {u1, v0}},
		{{x1, y1}, {}, {u1, v1}}, {{x0, y1}, {}, {u0, v1}}
	};
	DrawQuad({vertices, 4});

	// Restore even when the quad upload failed. The caller may have had blending disabled.
	m_d3d_context->PSSetShaderResources(0, 2, null_views);
	m_d3d_context->OMSetRenderTargets(1, &original_target, original_depth);
	m_current_blend_state = original_blend;
	if (original_blend_enabled) SetBlendState(original_blend);
	else DisableBlend();
	SetTargetSpace(original_width, original_height, original_origin, original_is_layer);
	Dx11CleanupResourceIfCreated(original_target);
	Dx11CleanupResourceIfCreated(original_depth);
}

void RenderInterface_DX11::BlitLayerToPostprocessPrimary(Rml::LayerHandle layer_handle, bool allow_regional, bool preserve_space)
{
	if (!preserve_space)
	{
		m_render_layers->SelectPostprocess(layer_handle);
		m_render_layers->SetPostprocessBounds(m_render_layers->GetLayer(layer_handle).bounds, 0);
	}
	const Gfx::RenderTargetData& source = m_render_layers->GetLayer(layer_handle);
	const Gfx::RenderTargetData& destination = m_render_layers->GetPostprocessPrimary();

	// Resolve directly into a cropped target; avoid a full-size intermediate for a small filter.
	if (allow_regional && source.sample_count == MsaaSamples && source.render_target_shader_resource_view && program_data &&
		program_data->programs[ProgramId::ResolveRegion].pixel_shader)
	{
		auto region = source.bounds.Intersect(destination.bounds);
		if (m_scissor_state.Valid())
			region = region.Intersect(m_scissor_state);
		const uint64_t pixels = uint64_t(Rml::Math::Max(0, region.Width())) * uint64_t(Rml::Math::Max(0, region.Height()));
		const uint64_t full_pixels = uint64_t(source.width) * uint64_t(source.height);
		D3D11_MAPPED_SUBRESOURCE mapping{};
		const bool cropped = source.width != destination.width || source.height != destination.height || source.bounds.p0 != destination.bounds.p0;
		if (pixels && (cropped || pixels <= full_pixels / 4) &&
			SUCCEEDED(m_d3d_context->Map(m_drop_shadow_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapping)))
		{
			const auto offset = destination.bounds.p0 - source.bounds.p0;
			const int values[4] = {offset.x, offset.y, 0, 0};
			memcpy(mapping.pData, values, sizeof(values));
			m_d3d_context->Unmap(m_drop_shadow_cbuffer, 0);
			ID3D11RenderTargetView* original_target = nullptr;
			ID3D11DepthStencilView* original_depth = nullptr;
			m_d3d_context->OMGetRenderTargets(1, &original_target, &original_depth);
			const int original_width = m_viewport_width, original_height = m_viewport_height;
			const auto original_origin = m_target_origin;
			const bool original_is_layer = m_target_is_layer;
			const auto original_scissor = m_scissor_state;
			const auto original_blend = m_current_blend_state;
			const bool original_blending = m_current_blend_state_enabled;
			ID3D11ShaderResourceView* null_views[2] = {};
			m_d3d_context->PSSetShaderResources(0, 2, null_views);
			const float clear[4] = {};
			m_d3d_context->ClearRenderTargetView(destination.render_target_view, clear);
			BindTarget(destination);
			BindTexture(source);
			DisableBlend();
			SetScissor(region);
			UseProgram(ProgramId::ResolveRegion);
			m_d3d_context->PSSetConstantBuffers(1, 1, &m_drop_shadow_cbuffer);
			DrawFullscreenQuad();
			m_d3d_context->PSSetShaderResources(0, 2, null_views);
			m_d3d_context->OMSetRenderTargets(1, &original_target, original_depth);
			if (original_blending)
				SetBlendState(original_blend);
			else DisableBlend();
			SetTargetSpace(original_width, original_height, original_origin, original_is_layer);
			SetScissor(original_scissor);
			Dx11CleanupResourceIfCreated(original_target);
			Dx11CleanupResourceIfCreated(original_depth);
			return;
		}
	}

	// Retile the resolved source into the filter's pixel-aligned working rectangle.
	ID3D11ShaderResourceView* null_views[2] = {};
	m_d3d_context->PSSetShaderResources(0, 2, null_views);
	const float clear[4] = {};
	m_d3d_context->ClearRenderTargetView(destination.render_target_view, clear);
	const auto offset = source.bounds.p0 - destination.bounds.p0;
	if (source.width == destination.width && source.height == destination.height && offset == Rml::Vector2i{})
	{
		BlitRenderTarget(source, destination, 0, 0, source.width, source.height, 0, 0, destination.width, destination.height);
		return;
	}
	const auto& resolved = m_render_layers->GetResolveTarget(source);
	m_d3d_context->OMSetRenderTargets(0, nullptr, nullptr);
	m_d3d_context->ResolveSubresource(resolved.render_target_texture, 0, source.render_target_texture, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
	const auto blend = m_current_blend_state;
	const bool blending = m_current_blend_state_enabled;
	BindTarget(destination);
	DisableBlend();
	BindTexture(resolved, 0, false);
	m_d3d_context->PSSetSamplers(0, 1, &m_postprocess_sampler_state);
	UseProgram(ProgramId::Passthrough);
	const Rml::Vector2f scale{float(destination.bounds.Width()) / source.width, float(destination.bounds.Height()) / source.height};
	DrawFullscreenQuad({float(-offset.x) / source.width, 1.f - scale.y + float(offset.y) / source.height}, scale);
	m_d3d_context->PSSetShaderResources(0, 2, null_views);
	if (blending)
		SetBlendState(blend);
	else
		DisableBlend();
}

enum class FilterType { Invalid = 0, Passthrough, Blur, DropShadow, ColorMatrix, MaskImage };
struct CompiledFilter {
	FilterType type;

	// Passthrough
	float blend_factor;

	// Blur
	float sigma;

	// Drop shadow
	Rml::Vector2f offset;
	Rml::ColourbPremultiplied color;

	// ColorMatrix
	Rml::Matrix4f color_matrix;
	Rml::SharedPtr<Gfx::RenderTargetData> mask;
	Rml::Rectanglei mask_bounds = Rml::Rectanglei::MakeInvalid();
};

static void SigmaToParameters(float desired_sigma, int& out_pass_level, float& out_sigma);

Rml::LayerHandle RenderInterface_DX11::PushLayer()
{
	return PushLayerWithBounds(Rml::Rectanglei::MakeInvalid());
}

Rml::LayerHandle RenderInterface_DX11::PushLayerWithBounds(Rml::Rectanglei bounds)
{
	if (bounds.Valid())
		bounds = RasterBounds(bounds).Extend(2).Intersect(Rml::Rectanglei::FromSize({m_canvas_width, m_canvas_height}));
	const auto layer = m_render_layers->PushLayer(bounds);
	BindTarget(m_render_layers->GetTopLayer());
	const float clear[4] = {};
	m_d3d_context->ClearRenderTargetView(m_render_layers->GetTopLayer().render_target_view, clear);
	return layer;
}

void RenderInterface_DX11::CompositeLayers(Rml::LayerHandle source_handle, Rml::LayerHandle destination_handle, Rml::BlendMode blend_mode,
	Rml::Span<const Rml::CompiledFilterHandle> filters)
{
	CompositeLayersInternal(source_handle, destination_handle, blend_mode, filters, m_scissor_state);
}

void RenderInterface_DX11::BeginTextureGeneration()
{
	++m_texture_generation_depth;
}
void RenderInterface_DX11::EndTextureGeneration()
{
	RMLUI_ASSERT(m_texture_generation_depth);
	--m_texture_generation_depth;
}

bool RenderInterface_DX11::SupportsCompositeLayersWithInputBounds() const
{
	return true;
}

void RenderInterface_DX11::CompositeLayersWithInputBounds(Rml::LayerHandle source_handle, Rml::LayerHandle destination_handle,
	Rml::BlendMode blend_mode, Rml::Span<const Rml::CompiledFilterHandle> filters, Rml::Rectanglei input_bounds)
{
	CompositeLayersInternal(source_handle, destination_handle, blend_mode, filters, RasterBounds(input_bounds));
}

void RenderInterface_DX11::CompositeLayersInternal(Rml::LayerHandle source_handle, Rml::LayerHandle destination_handle,
	Rml::BlendMode blend_mode, Rml::Span<const Rml::CompiledFilterHandle> filters, Rml::Rectanglei input_bounds)
{
	using Rml::BlendMode;
	const Rml::Rectanglei output_bounds = m_scissor_state;
	// Filtering needs the halo around the output, while the current stencil belongs only to the destination.
	SetScissor(input_bounds);

	// Blit source layer to postprocessing buffer. Do this regardless of whether we actually have any filters to be
	// applied, because we need to resolve the multi-sampled framebuffer in any case.
	// @performance If we have BlendMode::Replace and no filters or mask then we can just blit directly to the destination.
	m_render_layers->SelectPostprocess(source_handle);
	int alignment = 0;
	for (auto handle : filters)
	{
		const auto& filter = *reinterpret_cast<const CompiledFilter*>(handle);
		if (filter.type == FilterType::Blur || filter.type == FilterType::DropShadow)
		{
			int level = 0;
			float sigma = 0;
			SigmaToParameters(filter.sigma * Rml::Math::Max(m_raster_scale.x, m_raster_scale.y), level, sigma);
			alignment = Rml::Math::Max(alignment, 1 << level);
		}
	}
	// Callback generation already has a bounded viewport. Keep its established blur sampling grid.
	if (m_texture_generation_depth)
		alignment = 0;
	if (alignment)
		m_render_layers->SetPostprocessBounds(input_bounds, alignment);
	else
		m_render_layers->SetPostprocessBounds(m_render_layers->GetLayer(source_handle).bounds, 0);
	BlitLayerToPostprocessPrimary(source_handle, true, true);
	BindTarget(m_render_layers->GetPostprocessPrimary());
	if (m_backdrop_active && source_handle == 0 && m_backdrop_texture)
	{
		const auto& source = m_render_layers->GetPostprocessPrimary();
		const auto& destination = m_render_layers->GetPostprocessSecondary();
		BindTarget(destination);
		BindTexture(source);
		m_d3d_context->PSSetShaderResources(1, 1, &m_backdrop_texture);
		m_d3d_context->PSSetSamplers(0, 1, &m_postprocess_sampler_state);
		const auto blend = m_current_blend_state;
		const bool blending = m_current_blend_state_enabled;
		DisableBlend();
		UseProgram(ProgramId::SurfaceBackdrop);
		D3D11_MAPPED_SUBRESOURCE mapping{};
		if (FAILED(m_d3d_context->Map(m_drop_shadow_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapping)))
		{
			BindTarget(m_render_layers->GetTopLayer());
			SetScissor(output_bounds);
			return;
		}
		const float values[8] = {float(m_target_origin.x), float(m_target_origin.y), 1.f / m_canvas_width, 1.f / m_canvas_height};
		memcpy(mapping.pData, values, sizeof(values));
		m_d3d_context->Unmap(m_drop_shadow_cbuffer, 0);
		m_d3d_context->PSSetConstantBuffers(1, 1, &m_drop_shadow_cbuffer);
		DrawFullscreenQuad();
		ID3D11ShaderResourceView* nullView = nullptr;
		m_d3d_context->PSSetShaderResources(1, 1, &nullView);
		if (blending) SetBlendState(blend);
		m_render_layers->SwapPostprocessPrimarySecondary();
	}

	// Render the filters, the PostprocessPrimary framebuffer is used for both input and output.
	RenderFilters(filters);
	SetScissor(output_bounds);

	// Render to the destination layer.
	const auto& destination = m_render_layers->GetLayer(destination_handle);
	BindTarget(destination);
	EnsureClip(destination);
	BindTexture(m_render_layers->GetPostprocessPrimary());

	UseProgram(ProgramId::Passthrough);

	if (blend_mode == BlendMode::Replace)
		DisableBlend();

	DrawTargetQuad(m_render_layers->GetPostprocessPrimary(), destination);

	if (blend_mode == BlendMode::Replace)
		SetBlendState(m_current_blend_state);

	if (destination_handle != m_render_layers->GetTopLayerHandle())
		BindTarget(m_render_layers->GetTopLayer());
}

void RenderInterface_DX11::PopLayer()
{
	m_render_layers->PopLayer();
	BindTarget(m_render_layers->GetTopLayer());
}

Rml::TextureHandle RenderInterface_DX11::SaveLayerAsTexture()
{
	RMLUI_ASSERT(m_scissor_state.Valid());
	const Rml::Rectanglei bounds = m_scissor_state;

	Rml::TextureHandle render_texture = GenerateTexture({}, bounds.Size());
	if (!render_texture)
	{
		return {};
	}

	BlitLayerToPostprocessPrimary(m_render_layers->GetTopLayerHandle());

	EnableScissorRegion(false);

	const Gfx::RenderTargetData& source = m_render_layers->GetPostprocessPrimary();
	const Gfx::RenderTargetData& destination = m_render_layers->GetPostprocessSecondary();

	// Flip the image vertically, as that convention is used for textures, and move to origin.
	BlitRenderTarget(source, destination, bounds.Left() - source.bounds.Left(),
		source.bounds.Height() - (bounds.Bottom() - source.bounds.Top()),                                     // src0
		bounds.Right() - source.bounds.Left(), source.bounds.Height() - (bounds.Top() - source.bounds.Top()), // src1
		0, bounds.Height(),                                                                                   // dst0
		bounds.Width(), 0                                                                                     // dst1
	);

	// Now we need to copy the destination texture to the final texture for rendering
	// Bind the destination texture as the source for copying to the final texture (render_texture)
	// after extracting the associated resource with it
	ID3D11ShaderResourceView* texture_view = (ID3D11ShaderResourceView*)render_texture;
	ID3D11Resource* texture_resource = nullptr;
	texture_view->GetResource(&texture_resource);

	// Copy the destination texture to the final texture resource
	D3D11_BOX copy_box{};
	copy_box.left = 0;
	copy_box.right = bounds.Width();
	copy_box.top = 0;
	copy_box.bottom = bounds.Height();
	copy_box.front = 0;
	copy_box.back = 1;

	// Copy the blitted content from the destination to the final texture
	m_d3d_context->CopySubresourceRegion(texture_resource, 0, 0, 0, 0, destination.render_target_texture, 0, &copy_box);
	// GenerateTexture({}, ...) leaves the mip chain empty until mip 0 has been copied.
	// Minified saved layers (including cached box shadows/backgrounds) sample those levels.
	m_d3d_context->GenerateMips(texture_view);

	// Restore state and free memory
	SetScissor(bounds);
	BindTarget(m_render_layers->GetTopLayer());
	texture_resource->Release();

	return render_texture;
}

Rml::CompiledFilterHandle RenderInterface_DX11::SaveLayerAsMaskImage()
{
	BlitLayerToPostprocessPrimary(m_render_layers->GetTopLayerHandle());
	const auto& source = m_render_layers->GetPostprocessPrimary();
	Rml::SharedPtr<Gfx::RenderTargetData> image;
	for (auto& slot : m_mask_pool)
		if (slot.use_count() == 1)
		{
			image = slot;
			break;
		}
	if (!image)
	{
		image = Rml::SharedPtr<Gfx::RenderTargetData>(new Gfx::RenderTargetData, [](Gfx::RenderTargetData* value) {
			Gfx::DestroyRenderTarget(*value);
			delete value;
		});
		m_mask_pool.push_back(image);
	}
	if (image->width < source.bounds.Width() || image->height < source.bounds.Height())
	{
		const int width = Rml::Math::Max(image->width, source.bounds.Width()), height = Rml::Math::Max(image->height, source.bounds.Height());
		Gfx::DestroyRenderTarget(*image);
		if (!Gfx::CreateRenderTarget(m_d3d_device, *image, width, height, 0, Gfx::RenderTargetAttachment::None, nullptr))
			return {};
	}
	ID3D11ShaderResourceView* null_views[2] = {};
	m_d3d_context->PSSetShaderResources(0, 2, null_views);
	m_d3d_context->OMSetRenderTargets(0, nullptr, nullptr);
	D3D11_BOX box{0, 0, 0, UINT(source.bounds.Width()), UINT(source.bounds.Height()), 1};
	m_d3d_context->CopySubresourceRegion(image->render_target_texture, 0, 0, 0, 0, source.render_target_texture, 0, &box);
	BindTarget(m_render_layers->GetTopLayer());
	CompiledFilter filter = {};
	filter.type = FilterType::MaskImage;
	filter.mask = std::move(image);
	filter.mask_bounds = source.bounds;
	return reinterpret_cast<Rml::CompiledFilterHandle>(new CompiledFilter(std::move(filter)));
}

Rml::CompiledFilterHandle RenderInterface_DX11::CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters)
{
	CompiledFilter filter = {};

	if (name == "opacity")
	{
		filter.type = FilterType::Passthrough;
		filter.blend_factor = Rml::Get(parameters, "value", 1.0f);
	}
	else if (name == "blur")
	{
		filter.type = FilterType::Blur;
		filter.sigma = Rml::Get(parameters, "sigma", 1.0f);
	}
	else if (name == "drop-shadow")
	{
		filter.type = FilterType::DropShadow;
		filter.sigma = Rml::Get(parameters, "sigma", 0.f);
		filter.color = Rml::Get(parameters, "color", Rml::Colourb()).ToPremultiplied();
		filter.offset = Rml::Get(parameters, "offset", Rml::Vector2f(0.f));
	}
	else if (name == "brightness")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		filter.color_matrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
	}
	else if (name == "contrast")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float grayness = 0.5f - 0.5f * value;
		filter.color_matrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
		filter.color_matrix.SetColumn(3, Rml::Vector4f(grayness, grayness, grayness, 1.f));
	}
	else if (name == "invert")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Math::Clamp(Rml::Get(parameters, "value", 1.0f), 0.f, 1.f);
		const float inverted = 1.f - 2.f * value;
		filter.color_matrix = Rml::Matrix4f::Diag(inverted, inverted, inverted, 1.f);
		filter.color_matrix.SetColumn(3, Rml::Vector4f(value, value, value, 1.f));
	}
	else if (name == "grayscale")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float rev_value = 1.f - value;
		const Rml::Vector3f gray = value * Rml::Vector3f(0.2126f, 0.7152f, 0.0722f);
		// clang-format off
        filter.color_matrix = Rml::Matrix4f::FromRows(
            {gray.x + rev_value, gray.y,             gray.z,             0.f},
            {gray.x,             gray.y + rev_value, gray.z,             0.f},
            {gray.x,             gray.y,             gray.z + rev_value, 0.f},
            {0.f,                0.f,                0.f,                1.f}
        );
		// clang-format on
	}
	else if (name == "sepia")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float rev_value = 1.f - value;
		const Rml::Vector3f r_mix = value * Rml::Vector3f(0.393f, 0.769f, 0.189f);
		const Rml::Vector3f g_mix = value * Rml::Vector3f(0.349f, 0.686f, 0.168f);
		const Rml::Vector3f b_mix = value * Rml::Vector3f(0.272f, 0.534f, 0.131f);
		// clang-format off
        filter.color_matrix = Rml::Matrix4f::FromRows(
            {r_mix.x + rev_value, r_mix.y,             r_mix.z,             0.f},
            {g_mix.x,             g_mix.y + rev_value, g_mix.z,             0.f},
            {b_mix.x,             b_mix.y,             b_mix.z + rev_value, 0.f},
            {0.f,                 0.f,                 0.f,                 1.f}
        );
		// clang-format on
	}
	else if (name == "hue-rotate")
	{
		// Hue-rotation and saturation values based on: https://www.w3.org/TR/filter-effects-1/#attr-valuedef-type-huerotate
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float s = Rml::Math::Sin(value);
		const float c = Rml::Math::Cos(value);
		// clang-format off
        filter.color_matrix = Rml::Matrix4f::FromRows(
            {0.213f + 0.787f * c - 0.213f * s,  0.715f - 0.715f * c - 0.715f * s,  0.072f - 0.072f * c + 0.928f * s,  0.f},
            {0.213f - 0.213f * c + 0.143f * s,  0.715f + 0.285f * c + 0.140f * s,  0.072f - 0.072f * c - 0.283f * s,  0.f},
            {0.213f - 0.213f * c - 0.787f * s,  0.715f - 0.715f * c + 0.715f * s,  0.072f + 0.928f * c + 0.072f * s,  0.f},
            {0.f,                               0.f,                               0.f,                               1.f}
        );
		// clang-format on
	}
	else if (name == "saturate")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		// clang-format off
        filter.color_matrix = Rml::Matrix4f::FromRows(
            {0.213f + 0.787f * value,  0.715f - 0.715f * value,  0.072f - 0.072f * value,  0.f},
            {0.213f - 0.213f * value,  0.715f + 0.285f * value,  0.072f - 0.072f * value,  0.f},
            {0.213f - 0.213f * value,  0.715f - 0.715f * value,  0.072f + 0.928f * value,  0.f},
            {0.f,                      0.f,                      0.f,                      1.f}
        );
		// clang-format on
	}

	if (filter.type != FilterType::Invalid)
		return reinterpret_cast<Rml::CompiledFilterHandle>(new CompiledFilter(std::move(filter)));

	Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported filter type '%s'.", name.c_str());
	return {};
}

void RenderInterface_DX11::ReleaseFilter(Rml::CompiledFilterHandle filter)
{
	delete reinterpret_cast<CompiledFilter*>(filter);
}

enum class CompiledShaderType { Invalid = 0, Gradient, Creation };
struct CompiledShader {
	CompiledShaderType type;

	// Gradient
	ShaderGradientFunction gradient_function;
	Rml::Vector2f p;
	Rml::Vector2f v;
	Rml::Vector<float> stop_positions;
	Rml::Vector<Rml::Colourf> stop_colors;

	// Shader
	Rml::Vector2f dimensions;
};

Rml::CompiledShaderHandle RenderInterface_DX11::CompileShader(const Rml::String& name, const Rml::Dictionary& parameters)
{
	auto ApplyColorStopList = [](CompiledShader& shader, const Rml::Dictionary& shader_parameters) {
		auto it = shader_parameters.find("color_stop_list");
		RMLUI_ASSERT(it != shader_parameters.end() && it->second.GetType() == Rml::Variant::COLORSTOPLIST);
		const Rml::ColorStopList& color_stop_list = it->second.GetReference<Rml::ColorStopList>();
		const int num_stops = Rml::Math::Min((int)color_stop_list.size(), MAX_NUM_STOPS);

		shader.stop_positions.resize(num_stops);
		shader.stop_colors.resize(num_stops);
		for (int i = 0; i < num_stops; i++)
		{
			const Rml::ColorStop& stop = color_stop_list[i];
			RMLUI_ASSERT(stop.position.unit == Rml::Unit::NUMBER);
			shader.stop_positions[i] = stop.position.number;
			shader.stop_colors[i] = ConvertToColorf(stop.color);
		}
	};

	CompiledShader shader = {};

	if (name == "linear-gradient")
	{
		shader.type = CompiledShaderType::Gradient;
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.gradient_function = (repeating ? ShaderGradientFunction::RepeatingLinear : ShaderGradientFunction::Linear);
		shader.p = Rml::Get(parameters, "p0", Rml::Vector2f(0.f));
		shader.v = Rml::Get(parameters, "p1", Rml::Vector2f(0.f)) - shader.p;
		ApplyColorStopList(shader, parameters);
	}
	else if (name == "radial-gradient")
	{
		shader.type = CompiledShaderType::Gradient;
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.gradient_function = (repeating ? ShaderGradientFunction::RepeatingRadial : ShaderGradientFunction::Radial);
		shader.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
		shader.v = Rml::Vector2f(1.f) / Rml::Get(parameters, "radius", Rml::Vector2f(1.f));
		ApplyColorStopList(shader, parameters);
	}
	else if (name == "conic-gradient")
	{
		shader.type = CompiledShaderType::Gradient;
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.gradient_function = (repeating ? ShaderGradientFunction::RepeatingConic : ShaderGradientFunction::Conic);
		shader.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
		const float angle = Rml::Get(parameters, "angle", 0.f);
		shader.v = {Rml::Math::Cos(angle), Rml::Math::Sin(angle)};
		ApplyColorStopList(shader, parameters);
	}
	else if (name == "shader")
	{
		const Rml::String value = Rml::Get(parameters, "value", Rml::String());
		if (value == "creation")
		{
			shader.type = CompiledShaderType::Creation;
			shader.dimensions = Rml::Get(parameters, "dimensions", Rml::Vector2f(0.f));
		}
	}

	if (shader.type != CompiledShaderType::Invalid)
		return reinterpret_cast<Rml::CompiledShaderHandle>(new CompiledShader(std::move(shader)));

	Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported shader type '%s'.", name.c_str());
	return {};
}

void RenderInterface_DX11::RenderShader(Rml::CompiledShaderHandle shader_handle, Rml::CompiledGeometryHandle geometry_handle,
	Rml::Vector2f translation, Rml::TextureHandle /*texture*/)
{
	EnsureClip(m_render_layers->GetTopLayer());
	RMLUI_ASSERT(shader_handle && geometry_handle);
	const CompiledShader& shader = *reinterpret_cast<CompiledShader*>(shader_handle);
	const CompiledShaderType type = shader.type;

	DX11_GeometryData* geometry = (DX11_GeometryData*)geometry_handle;

	switch (type)
	{
	case CompiledShaderType::Gradient:
	{
		RMLUI_ASSERT(shader.stop_positions.size() == shader.stop_colors.size());
		const int num_stops = (int)shader.stop_positions.size();

		UseProgram(ProgramId::Gradient);

		D3D11_MAPPED_SUBRESOURCE mappedResource{};
		// Lock the constant buffer so it can be written to.
		HRESULT result = m_d3d_context->Map(m_shader_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
		if (FAILED(result))
		{
			return;
		}

		// Get a pointer to the data in the constant buffer.
		ShaderCbuffer* dataPtr = (ShaderCbuffer*)mappedResource.pData;

		// Copy the data to the GPU
		dataPtr->common.transform = m_transform;
		dataPtr->common.translation = translation;
		dataPtr->gradient.func = static_cast<int>(shader.gradient_function);
		dataPtr->gradient.p = shader.p;
		dataPtr->gradient.v = shader.v;
		dataPtr->gradient.num_stops = num_stops;
		// Reset stop positions and colours to 0
		memset(dataPtr->gradient.stop_positions, 0, sizeof(dataPtr->gradient.stop_positions));
		memset(dataPtr->gradient.stop_colors, 0, sizeof(dataPtr->gradient.stop_colors));
		// Copy to stop position and colours
		memcpy_s(&dataPtr->gradient.stop_positions, num_stops * sizeof(float), shader.stop_positions.data(), num_stops * sizeof(float));
		memcpy_s(&dataPtr->gradient.stop_colors, num_stops * sizeof(Rml::Vector4f), shader.stop_colors.data(), num_stops * sizeof(Rml::Vector4f));

		// Upload to the GPU.
		m_d3d_context->Unmap(m_shader_buffer, 0);
		// The custom upload used its own translation; the next ordinary draw must restore the cached common state.
		m_cbuffer_dirty = true;

		// Issue draw call
		m_d3d_context->IASetInputLayout(m_vertex_layout);
		m_d3d_context->VSSetConstantBuffers(0, 1, &m_shader_buffer);
		m_d3d_context->PSSetConstantBuffers(0, 1, &m_shader_buffer);
		uint32_t stride = sizeof(Rml::Vertex);
		uint32_t offset = 0;
		m_d3d_context->IASetVertexBuffers(0, 1, &geometry->vertex_buffer, &stride, &offset);
		m_d3d_context->IASetIndexBuffer(geometry->index_buffer, DXGI_FORMAT_R32_UINT, 0);
		m_d3d_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_d3d_context->DrawIndexed(static_cast<UINT>(geometry->index_count), 0, 0);
	}
	break;
	case CompiledShaderType::Creation:
	{
		const double time = Rml::GetSystemInterface()->GetElapsedTime();

		UseProgram(ProgramId::Creation);

		D3D11_MAPPED_SUBRESOURCE mappedResource{};
		// Lock the constant buffer so it can be written to.
		HRESULT result = m_d3d_context->Map(m_shader_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
		if (FAILED(result))
		{
			return;
		}

		// Get a pointer to the data in the constant buffer.
		ShaderCbuffer* dataPtr = (ShaderCbuffer*)mappedResource.pData;

		// Copy the data to the GPU
		dataPtr->common.transform = m_transform;
		dataPtr->common.translation = translation;
		dataPtr->creation.value = (float)time;
		dataPtr->creation.dimensions = shader.dimensions;

		// Upload to the GPU.
		m_d3d_context->Unmap(m_shader_buffer, 0);
		// The custom upload used its own translation; the next ordinary draw must restore the cached common state.
		m_cbuffer_dirty = true;

		// Issue draw call
		m_d3d_context->IASetInputLayout(m_vertex_layout);
		m_d3d_context->VSSetConstantBuffers(0, 1, &m_shader_buffer);
		m_d3d_context->PSSetConstantBuffers(0, 1, &m_shader_buffer);
		uint32_t stride = sizeof(Rml::Vertex);
		uint32_t offset = 0;
		m_d3d_context->IASetVertexBuffers(0, 1, &geometry->vertex_buffer, &stride, &offset);
		m_d3d_context->IASetIndexBuffer(geometry->index_buffer, DXGI_FORMAT_R32_UINT, 0);
		m_d3d_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_d3d_context->DrawIndexed(static_cast<UINT>(geometry->index_count), 0, 0);
	}
	break;
	case CompiledShaderType::Invalid:
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "Unhandled render shader %d.", (int)type);
	}
	break;
	}
}

void RenderInterface_DX11::ReleaseShader(Rml::CompiledShaderHandle shader_handle)
{
	delete reinterpret_cast<CompiledShader*>(shader_handle);
}

static void SigmaToParameters(const float desired_sigma, int& out_pass_level, float& out_sigma)
{
	constexpr int max_num_passes = 10;
	static_assert(max_num_passes < 31, "");
	constexpr float max_single_pass_sigma = 3.0f;
	out_pass_level = Rml::Math::Clamp(Rml::Math::Log2(int(desired_sigma * (2.f / max_single_pass_sigma))), 0, max_num_passes);
	out_sigma = Rml::Math::Clamp(desired_sigma / float(1 << out_pass_level), 0.0f, max_single_pass_sigma);
}

static void SetTexCoordLimits(Rml::Vector2f& p_tex_coord_min, Rml::Vector2f& p_tex_coord_max, Rml::Rectanglei rectangle_flipped,
	Rml::Vector2i framebuffer_size, int logical_height = 0)
{
	// Offset by half-texel values so that texture lookups are clamped to fragment centers, thereby avoiding color
	// bleeding from neighboring texels due to bilinear interpolation.
	const Rml::Vector2f min = (Rml::Vector2f(rectangle_flipped.p0) + Rml::Vector2f(0.5f)) / Rml::Vector2f(framebuffer_size);
	const Rml::Vector2f max = (Rml::Vector2f(rectangle_flipped.p1) - Rml::Vector2f(0.5f)) / Rml::Vector2f(framebuffer_size);

	p_tex_coord_min.x = min.x;
	p_tex_coord_min.y = min.y + (logical_height ? 1.f - float(logical_height) / framebuffer_size.y : 0.f);
	p_tex_coord_max.x = max.x;
	p_tex_coord_max.y = max.y + (logical_height ? 1.f - float(logical_height) / framebuffer_size.y : 0.f);
}

static void SetBlurWeights(Rml::Vector4f& p_weights, float sigma)
{
	constexpr int num_weights = BLUR_NUM_WEIGHTS;
	float weights[num_weights];
	float normalization = 0.0f;
	for (int i = 0; i < num_weights; i++)
	{
		if (Rml::Math::Absolute(sigma) < 0.1f)
			weights[i] = float(i == 0);
		else
			weights[i] = Rml::Math::Exp(-float(i * i) / (2.0f * sigma * sigma)) / (Rml::Math::SquareRoot(2.f * Rml::Math::RMLUI_PI) * sigma);

		normalization += (i == 0 ? 1.f : 2.0f) * weights[i];
	}
	for (float& weight : weights)
		weight /= normalization;

	p_weights.x = weights[0];
	p_weights.y = weights[1];
	p_weights.z = weights[2];
	p_weights.w = weights[3];
}

void RenderInterface_DX11::RenderBlur(Rml::Vector2f sigma, const Gfx::RenderTargetData& source_destination, const Gfx::RenderTargetData& temp,
	const Rml::Rectanglei window_flipped)
{
	RMLUI_ASSERT(&source_destination != &temp && source_destination.width == temp.width && source_destination.height == temp.height);
	RMLUI_ASSERT(window_flipped.Valid());
	const int width = source_destination.bounds.Width(), height = source_destination.bounds.Height();

	if (window_flipped.Top() == window_flipped.Bottom() || window_flipped.Left() == window_flipped.Right())
	{
		// In some cases the bounds are 0 width, so early exit since there's nothing to blur
		return;
	}

	int pass_level = 0;
	float reduced_sigma = 0;
	SigmaToParameters(Rml::Math::Max(sigma.x, sigma.y), pass_level, reduced_sigma);
	sigma.x = Rml::Math::Clamp(sigma.x / float(1 << pass_level), 0.f, 3.f);
	sigma.y = Rml::Math::Clamp(sigma.y / float(1 << pass_level), 0.f, 3.f);
	if (sigma.x == 0 && sigma.y == 0)
	{
		// If sigma is zero we don't have anything to blur, so early exit since blurring will have no effect on the final image
		return;
	}

	const Rml::Rectanglei original_scissor = m_scissor_state;
	auto restore_state = [&]()
	{
		SetScissor(original_scissor);
		ID3D11ShaderResourceView* null_view = nullptr;
		m_d3d_context->PSSetShaderResources(0, 1, &null_view);
		// The blur WRITE_DISCARD upload does not preserve the common transform/translation prefix.
		m_cbuffer_dirty = true;
	};

	// Begin by downscaling so that the blur pass can be done at a reduced resolution for large sigma.
	Rml::Rectanglei scissor = window_flipped;

	UseProgram(ProgramId::Passthrough);
	SetLocalScissor(scissor, true);

	// Downscale by iterative half-scaling with bilinear filtering, to reduce aliasing.
	D3D11_VIEWPORT d3dviewport{};
	d3dviewport.TopLeftX = 0;
	// Align the reduced viewport to the bottom edge, including the extra row of an odd-height target.
	d3dviewport.TopLeftY = static_cast<FLOAT>(height - height / 2);
	d3dviewport.Width = static_cast<FLOAT>(width / 2);
	d3dviewport.Height = static_cast<FLOAT>(height / 2);
	d3dviewport.MinDepth = 0.0f;
	d3dviewport.MaxDepth = 1.0f;
	m_d3d_context->RSSetViewports(1, &d3dviewport);

	// Scale UVs if we have even dimensions, such that texture fetches align perfectly between texels, thereby producing a 50% blend of
	// neighboring texels.
	const Rml::Vector2f uv_scaling = {(width % 2 == 1) ? (1.f - 1.f / float(width)) : 1.f, (height % 2 == 1) ? (1.f - 1.f / float(height)) : 1.f};

	for (int i = 0; i < pass_level; i++)
	{
		scissor.p0 = (scissor.p0 + Rml::Vector2i(1)) / 2;
		scissor.p1 = Rml::Math::Max(scissor.p1 / 2, scissor.p0);
		const bool from_source = (i % 2 == 0);
		BindTarget((from_source ? temp : source_destination), false);
		BindTexture(from_source ? source_destination : temp);
		SetLocalScissor(scissor, true);

		DrawFullscreenQuad({}, uv_scaling);
	}

	d3dviewport.TopLeftX = 0;
	d3dviewport.TopLeftY = 0;
	d3dviewport.Width = static_cast<FLOAT>(width);
	d3dviewport.Height = static_cast<FLOAT>(height);
	d3dviewport.MinDepth = 0.0f;
	d3dviewport.MaxDepth = 1.0f;
	m_d3d_context->RSSetViewports(1, &d3dviewport);

	// Ensure texture data end up in the temp buffer. Depending on the last downscaling, we might need to move it from the source_destination buffer.
	const bool transfer_to_temp_buffer = (pass_level % 2 == 0);
	if (transfer_to_temp_buffer)
	{
		BindTarget(temp, false);
		BindTexture(source_destination);
		DrawFullscreenQuad();
	}

	// Set up uniforms.
	const bool paired = program_data->programs[ProgramId::BlurPaired].pixel_shader != nullptr;
	UseProgram(paired ? ProgramId::BlurPaired : ProgramId::Blur);
	D3D11_MAPPED_SUBRESOURCE mappedResource{};
	// Lock the constant buffer so it can be written to.
	HRESULT result = m_d3d_context->Map(m_shader_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if (FAILED(result))
	{
		restore_state();
		return;
	}

	// Get a pointer to the data in the constant buffer.
	ShaderCbuffer* dataPtr = (ShaderCbuffer*)mappedResource.pData;

	const auto set_weights = [&](ShaderCbuffer::Blur& data, float axis_sigma)
	{
		SetBlurWeights(data.weights, axis_sigma);
		if (paired)
		{
			const auto weights = data.weights;
			const float inner = weights.x * .5f + weights.y, outer = weights.z + weights.w;
			data.paired_weights_offsets = {inner, outer, weights.y / inner, outer > 0 ? weights.w / outer : 0};
		}
	};
	set_weights(dataPtr->blur, sigma.y);
	SetTexCoordLimits(dataPtr->blur.texcoord_min, dataPtr->blur.texcoord_max, scissor, {source_destination.width, source_destination.height}, height);

	Rml::Vector2f texel_offset = Rml::Vector2f(0.f, 1.f) * (1.0f / float(temp.height));
	dataPtr->blur.texel_offset = texel_offset;
	const auto horizontal_parameters = dataPtr->blur;

	// Upload to the GPU.
	m_d3d_context->Unmap(m_shader_buffer, 0);
	// Unbind second pass
	ID3D11Buffer* null_buffer = nullptr;
	m_d3d_context->VSSetConstantBuffers(1, 1, &null_buffer);

	// Blur render pass - vertical.
	BindTarget(source_destination, false);
	BindTexture(temp);
	DrawFullscreenQuad();

	// Blur render pass - horizontal.
	BindTarget(temp, false);
	BindTexture(source_destination);

	// Add a 1px transparent border around the blur region by first clearing with a padded scissor. This helps prevent
	// artifacts when upscaling the blur result in the later step. On Intel and AMD, we have observed that during
	// blitting with linear filtering, pixels outside the 'src' region can be blended into the output. On the other
	// hand, it looks like Nvidia clamps the pixels to the source edge, which is what we really want. Regardless, we
	// work around the issue with this extra step.
	SetLocalScissor(scissor.Extend(1), true);
	// D3D11 does not support clearing based on scissor rects. So we use a D3D11Context1 to use ClearView, and define the D3D11_RECt here
	Rml::Rectanglei scissor_ext = scissor.Extend(1);
	scissor_ext = VerticallyFlipped(scissor_ext, m_viewport_height);

	// Some render APIs don't like offscreen positions (WebGL in particular), so clamp them to the viewport.
	const int x_min = Rml::Math::Clamp(scissor_ext.Left(), 0, m_viewport_width);
	const int y_min = Rml::Math::Clamp(scissor_ext.Top(), 0, m_viewport_height);
	const int x_max = Rml::Math::Clamp(scissor_ext.Right(), 0, m_viewport_width);
	const int y_max = Rml::Math::Clamp(scissor_ext.Bottom(), 0, m_viewport_height);

	D3D11_RECT rect_scissor = {};
	rect_scissor.left = x_min;
	rect_scissor.top = y_min;
	rect_scissor.right = x_max;
	rect_scissor.bottom = y_max;

	FLOAT clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	m_d3d_context->ClearView(temp.render_target_view, clear_color, &rect_scissor, 1);
	SetLocalScissor(scissor, true);

	// Update texel offset for upscale pass
	result = m_d3d_context->Map(m_blur_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if (FAILED(result))
	{
		restore_state();
		return;
	}

	ShaderCbuffer::Blur* dataPtrBlur = (ShaderCbuffer::Blur*)mappedResource.pData;
	*dataPtrBlur = horizontal_parameters;
	set_weights(*dataPtrBlur, sigma.x);

	texel_offset = Rml::Vector2f(1.f, 0.f) * (1.0f / float(source_destination.width));
	dataPtrBlur->texel_offset = texel_offset;

	// Upload to the GPU.
	m_d3d_context->Unmap(m_blur_cbuffer, 0);
	m_d3d_context->VSSetConstantBuffers(0, 1, &m_blur_cbuffer);
	m_d3d_context->PSSetConstantBuffers(0, 1, &m_blur_cbuffer);

	RenderGeometry(m_fullscreen_quad_geometry, {}, RenderInterface_DX11::TexturePostprocessNoBinding);

	// Blit the blurred image to the scissor region with upscaling.
	Rml::Rectanglei window_flipped_twice = VerticallyFlipped(window_flipped, m_viewport_height);
	SetLocalScissor(window_flipped_twice, false);

	const Rml::Vector2i src_min = scissor.p0;
	const Rml::Vector2i src_max = scissor.p1;
	const Rml::Vector2i dst_min = window_flipped_twice.p0;
	const Rml::Vector2i dst_max = window_flipped_twice.p1;
	BlitRenderTarget(temp, source_destination, src_min.x, src_min.y, src_max.x, src_max.y, dst_min.x, dst_max.y, dst_max.x, dst_min.y);

	// The above upscale blit might be jittery at low resolutions (large pass levels). This is especially noticeable when moving an element with
	// backdrop blur around or when trying to click/hover an element within a blurred region since it may be rendered at an offset. For more stable
	// and accurate rendering we next upscale the blur image by an exact power-of-two. However, this may not fill the edges completely so we need to
	// do the above first. Note that this strategy may sometimes result in visible seams. Alternatively, we could try to enlarge the window to the
	// next power-of-two size and then downsample and blur that.

	// We need to flip the scissor rect to have it match DX11 coords
	const Rml::Vector2i target_min = src_min * (1 << pass_level);
	const Rml::Vector2i target_max = src_max * (1 << pass_level);
	if (target_min != window_flipped.p0 || target_max != window_flipped.p1)
	{
		// Convert to DX coordinates only after scaling back to the full-resolution GL rectangle.
		window_flipped_twice = VerticallyFlipped(Rml::Rectanglei::FromCorners(target_min, target_max), m_viewport_height);
		const Rml::Vector2i target_min_blit = window_flipped_twice.p0;
		const Rml::Vector2i target_max_blit = window_flipped_twice.p1;
		BlitRenderTarget(temp, source_destination, src_min.x, src_min.y, src_max.x, src_max.y, target_min_blit.x, target_max_blit.y,
			target_max_blit.x, target_min_blit.y);
	}

	// Restore render state.
	restore_state();
}

void RenderInterface_DX11::RenderFilters(Rml::Span<const Rml::CompiledFilterHandle> filter_handles)
{
	m_d3d_context->PSSetSamplers(0, 1, &m_postprocess_sampler_state);

	// To unbind slots when necessary
	ID3D11ShaderResourceView* null_shader_resource_views[2] = {nullptr, nullptr};

	for (const Rml::CompiledFilterHandle filter_handle : filter_handles)
	{
		const CompiledFilter& filter = *reinterpret_cast<const CompiledFilter*>(filter_handle);
		const FilterType type = filter.type;

		switch (type)
		{
		case FilterType::Passthrough:
		{
			UseProgram(ProgramId::Passthrough);
			const float blend_factor[4] = {filter.blend_factor, filter.blend_factor, filter.blend_factor, filter.blend_factor};
			m_d3d_context->OMSetBlendState(m_blend_state_color_filter, blend_factor, 0xFFFFFFFF);
			m_current_blend_state = m_blend_state_color_filter;

			const Gfx::RenderTargetData& source = m_render_layers->GetPostprocessPrimary();
			const Gfx::RenderTargetData& destination = m_render_layers->GetPostprocessSecondary();
			BindTarget(destination);
			BindTexture(source);
			m_d3d_context->PSSetSamplers(0, 1, &m_postprocess_sampler_state);

			DrawFullscreenQuad();

			m_render_layers->SwapPostprocessPrimarySecondary();
			SetBlendState(m_blend_state_enable);
			m_d3d_context->PSSetShaderResources(0, 2, null_shader_resource_views);
		}
		break;
		case FilterType::Blur:
		{
			ID3D11BlendState* blend_state_backup = m_current_blend_state;
			DisableBlend();

			const Gfx::RenderTargetData& source_destination = m_render_layers->GetPostprocessPrimary();
			const Gfx::RenderTargetData& temp = m_render_layers->GetPostprocessSecondary();

			const Rml::Rectanglei window_flipped = VerticallyFlipped(LocalBounds(m_scissor_state), m_viewport_height);
			RenderBlur(m_raster_scale * filter.sigma, source_destination, temp, window_flipped);

			SetBlendState(blend_state_backup);
		}
		break;
		case FilterType::DropShadow:
		{
			UseProgram(ProgramId::DropShadow);
			ID3D11BlendState* blend_state_backup = m_current_blend_state;
			DisableBlend();

			const Gfx::RenderTargetData& primary = m_render_layers->GetPostprocessPrimary();
			const Gfx::RenderTargetData& secondary = m_render_layers->GetPostprocessSecondary();
			BindTarget(secondary);
			BindTexture(primary);

			D3D11_MAPPED_SUBRESOURCE mappedResource{};
			// Lock the constant buffer so it can be written to.
			HRESULT result = m_d3d_context->Map(m_drop_shadow_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
			if (FAILED(result))
			{
				return;
			}

			// Get a pointer to the data in the constant buffer.
			ShaderCbuffer::DropShadow* dataPtr = (ShaderCbuffer::DropShadow*)mappedResource.pData;

			Rml::Colourf color = ConvertToColorf(filter.color);

			// Copy the data to the GPU
			dataPtr->color.x = color.red;
			dataPtr->color.y = color.green;
			dataPtr->color.z = color.blue;
			dataPtr->color.w = color.alpha;

			const Rml::Rectanglei window_flipped = VerticallyFlipped(LocalBounds(m_scissor_state), m_viewport_height);
			SetTexCoordLimits(dataPtr->texcoord_min, dataPtr->texcoord_max, LocalBounds(m_scissor_state), {primary.width, primary.height});

			// Upload to the GPU.
			m_d3d_context->Unmap(m_drop_shadow_cbuffer, 0);
			m_d3d_context->PSSetConstantBuffers(1, 1, &m_drop_shadow_cbuffer);

			m_cbuffer_dirty = true;

			const Rml::Vector2f uv_offset = (filter.offset * m_raster_scale) / Rml::Vector2f(-(float)m_viewport_width, (float)m_viewport_height);
			DrawFullscreenQuad(uv_offset);

			if (filter.sigma >= 0.5f)
			{
				const Gfx::RenderTargetData& tertiary = m_render_layers->GetPostprocessTertiary();
				RenderBlur(m_raster_scale * filter.sigma, secondary, tertiary, window_flipped);
			}

			UseProgram(ProgramId::Passthrough);
			BindTarget(secondary);
			BindTexture(primary);
			SetBlendState(blend_state_backup);
			DrawFullscreenQuad();

			m_render_layers->SwapPostprocessPrimarySecondary();

			ID3D11ShaderResourceView* const nullSRV = nullptr;
			m_d3d_context->PSSetShaderResources(0, 1, &nullSRV);
			m_d3d_context->PSSetShaderResources(1, 1, &nullSRV);
		}
		break;
		case FilterType::ColorMatrix:
		{
			UseProgram(ProgramId::ColorMatrix);

			ID3D11BlendState* blend_state_backup = m_current_blend_state;
			DisableBlend();

			const Gfx::RenderTargetData& source = m_render_layers->GetPostprocessPrimary();
			const Gfx::RenderTargetData& destination = m_render_layers->GetPostprocessSecondary();
			BindTarget(destination);
			BindTexture(source);

			D3D11_MAPPED_SUBRESOURCE mappedResource{};
			// Lock the constant buffer so it can be written to.
			HRESULT result = m_d3d_context->Map(m_color_matrix_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
			if (FAILED(result))
			{
				// Restore blend state
				SetBlendState(blend_state_backup);

				ID3D11ShaderResourceView* const nullSRV = nullptr;
				m_d3d_context->PSSetShaderResources(0, 1, &nullSRV);
				m_cbuffer_dirty = true;
				return;
			}

			// Get a pointer to the data in the constant buffer.
			ColorMatrixCbuffer* dataPtr = (ColorMatrixCbuffer*)mappedResource.pData;

			constexpr bool transpose = std::is_same<decltype(filter.color_matrix), Rml::RowMajorMatrix4f>::value;

			// Copy the data to the GPU
			memcpy_s(&dataPtr->color_matrix, sizeof(dataPtr->color_matrix),
				transpose ? filter.color_matrix.Transpose().data() : filter.color_matrix.data(), sizeof(dataPtr->color_matrix));

			// Upload to the GPU.
			m_d3d_context->Unmap(m_color_matrix_cbuffer, 0);

			m_d3d_context->PSSetConstantBuffers(1, 1, &m_color_matrix_cbuffer);

			DrawFullscreenQuad();

			// Unbind cbuffer
			ID3D11Buffer* null_cbuffer = nullptr;
			m_d3d_context->PSSetConstantBuffers(1, 1, &null_cbuffer);

			m_render_layers->SwapPostprocessPrimarySecondary();

			// Restore blend state
			SetBlendState(blend_state_backup);

			ID3D11ShaderResourceView* const nullSRV = nullptr;
			m_d3d_context->PSSetShaderResources(0, 1, &nullSRV);
			m_cbuffer_dirty = true;
		}
		break;
		case FilterType::MaskImage:
		{
			UseProgram(ProgramId::BlendMask);
			ID3D11BlendState* blend_state_backup = m_current_blend_state;
			DisableBlend();

			const Gfx::RenderTargetData& source = m_render_layers->GetPostprocessPrimary();
			const Gfx::RenderTargetData& blend_mask = *filter.mask;
			D3D11_MAPPED_SUBRESOURCE mapping{};
			if (FAILED(m_d3d_context->Map(m_drop_shadow_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapping)))
				return;
			const auto offset = m_target_origin - filter.mask_bounds.p0;
			const float values[8] = {float(offset.x), float(offset.y), 1.f / blend_mask.width, 1.f / blend_mask.height,
				float(filter.mask_bounds.Width()) / blend_mask.width, float(filter.mask_bounds.Height()) / blend_mask.height, 0, 0};
			memcpy(mapping.pData, values, sizeof(values));
			m_d3d_context->Unmap(m_drop_shadow_cbuffer, 0);
			m_d3d_context->PSSetConstantBuffers(1, 1, &m_drop_shadow_cbuffer);
			const Gfx::RenderTargetData& destination = m_render_layers->GetPostprocessSecondary();

			BindTarget(destination);
			BindTexture(source, 0);
			BindTexture(blend_mask, 1);

			DrawFullscreenQuad();

			m_render_layers->SwapPostprocessPrimarySecondary();
			SetBlendState(blend_state_backup);
			m_d3d_context->PSSetShaderResources(0, 2, null_shader_resource_views);
		}
		break;
		case FilterType::Invalid:
		{
			Rml::Log::Message(Rml::Log::LT_WARNING, "Unhandled render filter %d.", (int)type);
		}
		break;
		}

		// Unbind resources
		ID3D11ShaderResourceView* const nullSRV = nullptr;
		m_d3d_context->PSSetShaderResources(0, 1, &nullSRV);
	}
}

void RenderInterface_DX11::UpdateConstantBuffer()
{
	if (m_cbuffer_dirty)
	{
		D3D11_MAPPED_SUBRESOURCE mappedResource{};
		// Lock the constant buffer so it can be written to.
		HRESULT result = m_d3d_context->Map(m_shader_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
		if (FAILED(result))
		{
			return;
		}

		// Get a pointer to the data in the constant buffer.
		ShaderCbuffer* dataPtr = (ShaderCbuffer*)mappedResource.pData;

		// Copy the data to the GPU
		dataPtr->common.transform = m_transform;
		dataPtr->common.translation = m_translation;

		// Upload to the GPU.
		m_d3d_context->Unmap(m_shader_buffer, 0);
		m_cbuffer_dirty = false;
	}
}

void RenderInterface_DX11::UseProgram(ProgramId program_id)
{
	RMLUI_ASSERT(program_data);
	if (active_program != program_id)
	{
		if (program_id != ProgramId::None)
		{
			m_d3d_context->VSSetShader(program_data->programs[program_id].vertex_shader, nullptr, 0);
			m_d3d_context->PSSetShader(program_data->programs[program_id].pixel_shader, nullptr, 0);
		}
		active_program = program_id;
	}
}

void RenderInterface_DX11::EnableClipMask(bool enable)
{
	m_is_stencil_enabled = enable;
	if (enable)
	{
		m_d3d_context->OMSetDepthStencilState(m_depth_stencil_state_stencil_test, m_stencil_ref_value);
	}
	else
	{
		m_d3d_context->OMSetDepthStencilState(m_depth_stencil_state_disable, 0);
	}
}

void RenderInterface_DX11::DrawClip(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation)
{
	RMLUI_ASSERT(m_is_stencil_enabled);
	using Rml::ClipMaskOperation;

	ID3D11DepthStencilState* stencil_state = m_depth_stencil_state_disable;

	UINT stencil_test_value = m_stencil_ref_value;
	switch (operation)
	{
	case ClipMaskOperation::Set:
	{
		stencil_test_value = 1;
		stencil_state = m_depth_stencil_state_stencil_set;
	}
	break;
	case ClipMaskOperation::SetInverse:
	{
		stencil_test_value = 0;
		stencil_state = m_depth_stencil_state_stencil_set;
	}
	break;
	case ClipMaskOperation::Intersect:
	{
		stencil_test_value += 1;
		stencil_state = m_depth_stencil_state_stencil_intersect;
	}
	break;
	}

	// Disable writing to the color of the render target
	float blendFactor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
	m_d3d_context->OMSetBlendState(m_blend_state_disable_color, blendFactor, 0xffffffff);

	m_d3d_context->OMSetDepthStencilState(stencil_state, 1);

	bool clear_stencil = (operation == ClipMaskOperation::Set || operation == ClipMaskOperation::SetInverse);
	if (clear_stencil)
	{
		// Clear stencil buffer
		Rml::LayerHandle layer_handle = m_render_layers->GetTopLayerHandle();
		const Gfx::RenderTargetData& rtData = m_render_layers->GetLayer(layer_handle);
		ID3D11DepthStencilView* depth = nullptr;
		m_d3d_context->OMGetRenderTargets(0, nullptr, &depth);
		if (depth)
		{
			m_d3d_context->ClearDepthStencilView(depth, D3D11_CLEAR_STENCIL, 1, 0);
			depth->Release();
		}
	}

	RenderGeometry(geometry, translation, {});

	// Restore blend state
	m_d3d_context->OMSetBlendState(m_current_blend_state, blendFactor, 0xffffffff);
	// Restore state
	m_stencil_ref_value = stencil_test_value;
	m_d3d_context->OMSetDepthStencilState(m_depth_stencil_state_stencil_test, m_stencil_ref_value);
}

void RenderInterface_DX11::EnsureClip(const Gfx::RenderTargetData& target)
{
	if (!m_is_stencil_enabled || m_drawing_clip || !target.stencil)
		return;
	auto& stencil = *target.stencil;
	if (stencil.generation == m_clip_generation && stencil.origin == target.bounds.p0 && stencil.size == target.bounds.Size())
		return;
	const auto transform = m_user_transform;
	const auto scissor = m_scissor_state;
	m_drawing_clip = true;
	m_stencil_ref_value = 0;
	for (const auto& command : m_clip_commands)
	{
		SetTransform(&command.transform);
		SetScissor(command.scissor);
		DX11_GeometryData geometry{command.vertices.Get(), command.indices.Get(), command.index_count};
		DrawClip(command.operation, reinterpret_cast<Rml::CompiledGeometryHandle>(&geometry), command.translation);
	}
	SetTransform(&transform);
	SetScissor(scissor);
	m_drawing_clip = false;
	stencil.generation = m_clip_generation;
	stencil.origin = target.bounds.p0;
	stencil.size = target.bounds.Size();
	m_d3d_context->OMSetDepthStencilState(m_depth_stencil_state_stencil_test, m_stencil_ref_value);
}

void RenderInterface_DX11::RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle handle, Rml::Vector2f translation)
{
	if (operation == Rml::ClipMaskOperation::Intersect)
		EnsureClip(m_render_layers->GetTopLayer());
	else
		m_clip_commands.clear();
	const auto& geometry = *reinterpret_cast<DX11_GeometryData*>(handle);
	ClipCommand command{operation, geometry.vertex_buffer, geometry.index_buffer, geometry.index_count, m_user_transform, translation,
		m_scissor_state};
	m_clip_commands.push_back(std::move(command));
	++m_clip_generation;
	m_drawing_clip = true;
	DrawClip(operation, handle, translation);
	m_drawing_clip = false;
	if (auto stencil = m_render_layers->GetTopLayer().stencil)
	{
		stencil->generation = m_clip_generation;
		stencil->origin = m_target_origin;
		stencil->size = m_render_layers->GetTopLayer().bounds.Size();
	}
}

struct RenderInterface_DX11::SurfaceTarget
{
	Gfx::RenderTargetData color;
	~SurfaceTarget() { Gfx::DestroyRenderTarget(color); }
};

RenderInterface_DX11::SurfaceTarget* RenderInterface_DX11::CreateSurfaceTarget()
{
	return new SurfaceTarget;
}

void RenderInterface_DX11::ReleaseSurfaceTarget(SurfaceTarget* target) { delete target; }
void RenderInterface_DX11::ResetSurfaceScratch()
{
	m_surface_render_layers.Reset();
	m_mask_pool.clear();
}

void RenderInterface_DX11::SetSurfaceTarget(SurfaceTarget* target)
{
	m_surface_target = target;
	m_render_layers = target ? &m_surface_render_layers : &m_default_render_layers;
}

ID3D11RenderTargetView* RenderInterface_DX11::GetCurrentRenderTarget() const
{
	return m_render_layers->GetTopLayer().render_target_view;
}

ID3D11ShaderResourceView* RenderInterface_DX11::GetCurrentLayerView() const
{
	return m_render_layers->GetTopLayer().render_target_shader_resource_view;
}

ID3D11ShaderResourceView* RenderInterface_DX11::EndFrameTexture(ID3D11ShaderResourceView* foreground)
{
	m_render_layers->SelectPostprocess(0);
	const auto& active = m_render_layers->GetTopLayer();
	if (m_surface_target && (m_surface_target->color.width != m_canvas_width || m_surface_target->color.height != m_canvas_height))
	{
		Gfx::DestroyRenderTarget(m_surface_target->color);
		if (!Gfx::CreateRenderTarget(m_d3d_device, m_surface_target->color, m_canvas_width, m_canvas_height, 0, Gfx::RenderTargetAttachment::None,
				nullptr))
		{
			FinishFrame();
			return nullptr;
		}
	}
	const auto& resolved = m_surface_target ? m_surface_target->color : m_render_layers->GetResolveTarget(active);
	if (foreground && SupportsForegroundCache())
	{
		ID3D11ShaderResourceView* null_views[2] = {};
		m_d3d_context->PSSetShaderResources(0, 2, null_views);
		BindTarget(resolved);
		BindTexture(active);
		m_d3d_context->PSSetShaderResources(1, 1, &foreground);
		EnableClipMask(false);
		SetScissor(Rml::Rectanglei::MakeInvalid());
		DisableBlend();
		UseProgram(ProgramId::ForegroundResolve);
		DrawFullscreenQuad();
		m_d3d_context->PSSetShaderResources(0, 2, null_views);
	}
	else if (active.width == resolved.width && active.height == resolved.height)
		m_d3d_context->ResolveSubresource(resolved.render_target_texture, 0, active.render_target_texture, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
	else
	{
		const auto& scratch = m_render_layers->GetResolveTarget(active);
		m_d3d_context->ResolveSubresource(scratch.render_target_texture, 0, active.render_target_texture, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
		D3D11_BOX box{0, 0, 0, UINT(resolved.width), UINT(resolved.height), 1};
		m_d3d_context->OMSetRenderTargets(0, nullptr, nullptr);
		m_d3d_context->CopySubresourceRegion(resolved.render_target_texture, 0, 0, 0, 0, scratch.render_target_texture, 0, &box);
	}
	auto view = resolved.render_target_shader_resource_view;
	FinishFrame();
	return view;
}

bool RenderInterface_DX11::SupportsForegroundCache() const
{
	return program_data && program_data->programs[ProgramId::ForegroundResolve].pixel_shader;
}

ID3D11ShaderResourceView* RenderInterface_DX11::SaveCurrentLayerMsaa()
{
	const auto& source = m_render_layers->GetTopLayer();
	D3D11_TEXTURE2D_DESC desc{};
	source.render_target_texture->GetDesc(&desc);
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	ID3D11Texture2D* snapshot = nullptr;
	if (FAILED(m_d3d_device->CreateTexture2D(&desc, nullptr, &snapshot))) return nullptr;
	ID3D11ShaderResourceView* view = nullptr;
	const HRESULT hr = m_d3d_device->CreateShaderResourceView(snapshot, nullptr, &view);
	if (SUCCEEDED(hr))
	{
		m_d3d_context->OMSetRenderTargets(0, nullptr, nullptr);
		m_d3d_context->CopyResource(snapshot, source.render_target_texture);
		BindTarget(source);
	}
	snapshot->Release();
	return view;
}

RenderInterface_DX11::RenderLayerStack::RenderLayerStack() {}

RenderInterface_DX11::RenderLayerStack::~RenderLayerStack()
{
	DestroyRenderTargets();
}

void RenderInterface_DX11::RenderLayerStack::SetD3DResources(ID3D11Device* device)
{
	RMLUI_ASSERTMSG(!m_d3d_device, "D3D11Device has already been set!");
	m_d3d_device = device;
}

Rml::LayerHandle RenderInterface_DX11::RenderLayerStack::PushLayer(Rml::Rectanglei bounds)
{
	if (!bounds.Valid())
		bounds = Rml::Rectanglei::FromSize({m_width, m_height});
	bounds = Rml::Rectanglei::FromPositionSize(bounds.p0, Rml::Math::Max(bounds.Size(), Rml::Vector2i(1)));
	if (m_layers_size == int(m_rt_layers.size()))
	{
		m_rt_layers.emplace_back();
		m_rt_postprocess.emplace_back(6);
		m_postprocess_bounds.emplace_back();
		m_postprocess_capacity.emplace_back();
	}
	auto& layer = m_rt_layers[m_layers_size];
	const int width = Rml::Math::Max(1, Rml::Math::Max(layer.width, bounds.Width()));
	const int height = Rml::Math::Max(1, Rml::Math::Max(layer.height, bounds.Height()));
	if (!layer.render_target_view || layer.width != width || layer.height != height)
	{
		Gfx::DestroyRenderTarget(layer);
		const Gfx::RenderTargetData* shared = nullptr;
		for (const auto& other : m_rt_layers)
			if (&other != &layer && other.width == width && other.height == height && other.depth_stencil_view)
			{
				shared = &other;
				break;
			}
		Gfx::CreateRenderTarget(m_d3d_device, layer, width, height, MsaaSamples, Gfx::RenderTargetAttachment::DepthStencil,
			shared ? shared->depth_stencil_view : nullptr);
		if (shared)
			layer.stencil = shared->stencil;
	}
	layer.bounds = bounds;
	++m_layers_size;
	return GetTopLayerHandle();
}

void RenderInterface_DX11::RenderLayerStack::PopLayer()
{
	RMLUI_ASSERT(m_layers_size > 0);
	m_layers_size -= 1;
}

const Gfx::RenderTargetData& RenderInterface_DX11::RenderLayerStack::GetLayer(Rml::LayerHandle layer) const
{
	RMLUI_ASSERT((size_t)layer < (size_t)m_layers_size);
	return m_rt_layers[layer];
}

const Gfx::RenderTargetData& RenderInterface_DX11::RenderLayerStack::GetTopLayer() const
{
	return GetLayer(GetTopLayerHandle());
}

Rml::LayerHandle RenderInterface_DX11::RenderLayerStack::GetTopLayerHandle() const
{
	RMLUI_ASSERT(m_layers_size > 0);
	return static_cast<Rml::LayerHandle>(m_layers_size - 1);
}

void RenderInterface_DX11::RenderLayerStack::SwapPostprocessPrimarySecondary()
{
	std::swap(m_rt_postprocess[m_postprocess_layer][0], m_rt_postprocess[m_postprocess_layer][1]);
}

void RenderInterface_DX11::RenderLayerStack::BeginFrame(int new_width, int new_height)
{
	RMLUI_ASSERT(m_layers_size == 0);
	m_width = new_width;
	m_height = new_height;
	m_postprocess_layer = 0;
	PushLayer();
}

void RenderInterface_DX11::RenderLayerStack::EndFrame()
{
	RMLUI_ASSERT(m_layers_size == 1);
	PopLayer();
}

void RenderInterface_DX11::RenderLayerStack::DestroyRenderTargets()
{
	RMLUI_ASSERTMSG(m_layers_size == 0, "Do not call this during frame rendering, that is, between BeginFrame() and EndFrame().");

	for (Gfx::RenderTargetData& fb : m_rt_layers)
		Gfx::DestroyRenderTarget(fb);

	m_rt_layers.clear();

	for (auto& level : m_rt_postprocess)
		for (auto& fb : level)
			Gfx::DestroyRenderTarget(fb);
	m_rt_postprocess.clear();
	m_postprocess_bounds.clear();
	m_postprocess_capacity.clear();
}

void RenderInterface_DX11::RenderLayerStack::SelectPostprocess(Rml::LayerHandle layer)
{
	m_postprocess_layer = int(layer);
}

void RenderInterface_DX11::RenderLayerStack::SetPostprocessBounds(Rml::Rectanglei bounds, int alignment)
{
	const auto& layer = m_rt_layers[m_postprocess_layer];
	const auto canvas = Rml::Rectanglei::FromSize({m_width, m_height});
	if (alignment && bounds.Valid())
	{
		bounds = bounds.Extend(2);
		if (!canvas.Contains(bounds.p0) || !canvas.Contains(bounds.p1))
			bounds = canvas;
		else
		{
			const Rml::Vector2i origin{int(std::floor(double(bounds.Left()) / alignment)) * alignment,
				int(std::floor(double(bounds.Top() - m_height) / alignment)) * alignment + m_height};
			auto size = bounds.p1 - origin;
			size = ((size + Rml::Vector2i(alignment - 1)) / alignment) * alignment;
			bounds = Rml::Rectanglei::FromPositionSize(origin, size);
			if (!canvas.Contains(bounds.p0) || !canvas.Contains(bounds.p1))
				bounds = canvas;
		}
	}
	else
		bounds = layer.bounds;
	bounds = Rml::Rectanglei::FromPositionSize(bounds.p0, Rml::Math::Max(bounds.Size(), Rml::Vector2i(1)));
	auto& capacity = m_postprocess_capacity[m_postprocess_layer];
	capacity = Rml::Math::Max(capacity, bounds.Size());
	m_postprocess_bounds[m_postprocess_layer] = bounds;
}

const Gfx::RenderTargetData& RenderInterface_DX11::RenderLayerStack::EnsureRenderTargetPostprocess(int index)
{
	const auto size = m_postprocess_capacity[m_postprocess_layer];
	auto& rt = m_rt_postprocess[m_postprocess_layer][index];
	if (rt.width != size.x || rt.height != size.y)
	{
		Gfx::DestroyRenderTarget(rt);
		Gfx::CreateRenderTarget(m_d3d_device, rt, size.x, size.y, 0, Gfx::RenderTargetAttachment::None, nullptr);
	}
	rt.bounds = m_postprocess_bounds[m_postprocess_layer];
	return rt;
}

const Gfx::RenderTargetData& RenderInterface_DX11::RenderLayerStack::GetResolveTarget(const Gfx::RenderTargetData& source)
{
	auto& rt = m_rt_postprocess[m_postprocess_layer][5];
	if (rt.width != source.width || rt.height != source.height)
	{
		Gfx::DestroyRenderTarget(rt);
		Gfx::CreateRenderTarget(m_d3d_device, rt, source.width, source.height, 0, Gfx::RenderTargetAttachment::None, nullptr);
	}
	rt.bounds = source.bounds;
	return rt;
}
