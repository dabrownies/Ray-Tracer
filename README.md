# Ray Tracer

A physically-based ray tracing engine built from scratch in Python. Shoots virtual rays through each pixel to simulate how light actually behaves in the real world.

## What It Does

This ray tracer implements the core physics of light transport to create realistic images. Instead of using rasterization like most real-time graphics, it traces the path of light rays as they bounce around the scene.

### Features

- **Reflections** - Mirror surfaces that show other objects
- **Refractions** - Glass and water that bend light realistically using Snell's law
- **Shadows** - Proper occlusion from light sources
- **Anti-aliasing** - Smooth edges by sampling multiple rays per pixel
- **Texture mapping** - Patterns and images on surfaces
- **Global illumination** - Light bounces between objects for realistic lighting
- **Multiple materials** - Different surface types (matte, shiny, mirror, glass)
- **Multiple lights** - Colored light sources with distance falloff

## How It Works

1. **Camera setup** - Define viewpoint and field of view
2. **Ray casting** - Shoot rays through each pixel into the scene
3. **Intersection testing** - Find where rays hit objects (spheres, planes)
4. **Lighting calculation** - Compute how light sources illuminate each point
5. **Recursive bouncing** - Follow reflected and refracted rays for realism
6. **Color accumulation** - Blend all the light contributions together

The core algorithm traces rays recursively - when a ray hits a mirror, it spawns a reflection ray. When it hits glass, it spawns both reflection and refraction rays. This creates realistic light transport.

## Physics Implemented

- **Lambert's cosine law** - Diffuse surface scattering
- **Phong reflection model** - Specular highlights
- **Snell's law** - Refraction through transparent materials
- **Fresnel equations** - How much light reflects vs refracts at interfaces
- **Beer's law** - Light absorption through materials
- **Inverse square falloff** - Realistic light attenuation

## Getting Started

### Requirements

```bash
pip install numpy pillow
```

### Quick Test

```bash
python test.py
```

Creates a simple test image in about 10 seconds.

### Full Demo

```bash
python raytracer.py
```

Renders a complex scene with multiple materials and effects. Takes 2-5 minutes depending on your machine.

## Customizing Scenes

### Adding Objects

```python
# materials
red_matte = Material(Vec3(0.8, 0.2, 0.2), diffuse=0.8)
mirror = Material(Vec3(0.9, 0.9, 0.9), reflectivity=0.8)
glass = Material(Vec3(0.95, 0.95, 0.95), transparency=0.9, ior=1.5)

# objects
scene.add_object(Sphere(Vec3(0, 0, -3), 1.0, red_matte))
scene.add_object(Sphere(Vec3(2, 0, -3), 1.0, glass))
```

### Lighting

```python
# Different colored lights
scene.add_light(Light(Vec3(2, 4, -1), Vec3(1, 1, 1), intensity=1.0))      # White
scene.add_light(Light(Vec3(-2, 3, -2), Vec3(1, 0.5, 0.2), intensity=0.7)) # Orange
```

### Camera Angles

```python
camera = Camera(
    position=Vec3(0, 2, 5),    # Where camera is
    target=Vec3(0, 0, 0),      # What it's looking at
    up=Vec3(0, 1, 0),          # Which way is up
    fov=45,                    # Field of view
    aspect_ratio=16/9
)
```

## Performance Tuning

### Quality vs Speed

```python
# Fast preview
ray_tracer = RayTracer(max_depth=3, samples_per_pixel=1)

# Balanced quality
ray_tracer = RayTracer(max_depth=6, samples_per_pixel=4)

# High quality (slow)
ray_tracer = RayTracer(max_depth=10, samples_per_pixel=16)
```

### Image Sizes

- **Preview**: 200x200 (~5 seconds)
- **Demo**: 800x450 (~2-5 minutes)  
- **High res**: 1920x1080 (~30-60 minutes)

## Technical Details

### Ray-Object Intersection

Uses analytical solutions for geometric primitives:
- **Spheres** - Solve quadratic equation for ray-sphere intersection
- **Planes** - Compute ray-plane intersection using dot products

### Material Properties

Each material has physical parameters:
- **Albedo** - Base color/reflectance
- **Roughness** - How scattered the reflections are
- **Metallic** - Conductor vs dielectric behavior
- **IOR** - Index of refraction for transparent materials

### Anti-Aliasing

Shoots multiple rays per pixel with random jittering to smooth out jagged edges. Similar to supersampling but more efficient.

### Global Illumination

Implements recursive ray tracing where light bounces between surfaces multiple times. This creates realistic indirect lighting and color bleeding.

## Extending the Engine

Easy to add new features:
- **New primitives** - Triangles, boxes, cylinders
- **Procedural textures** - Noise, gradients, patterns  
- **Advanced materials** - Subsurface scattering, emission
- **Camera effects** - Depth of field, motion blur
- **Optimization** - BVH acceleration, GPU compute

## License

MIT License - Feel free to use and modify
