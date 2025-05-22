# Save this as test_raytracer.py
from raytracer import *

def quick_test():
    print("Starting quick ray tracer test...")
    
    # Create a simple scene
    scene = Scene()
    
    # Add one sphere
    red_material = Material(Vec3(1, 0, 0), diffuse=0.8)
    scene.add_object(Sphere(Vec3(0, 0, -3), 1, red_material))
    
    # Add one light
    scene.add_light(Light(Vec3(2, 2, 0), Vec3(1, 1, 1)))
    
    # Simple camera
    camera = Camera(
        position=Vec3(0, 0, 0),
        target=Vec3(0, 0, -3),
        up=Vec3(0, 1, 0),
        fov=45,
        aspect_ratio=1.0
    )
    
    # Fast ray tracer (low quality for speed)
    ray_tracer = RayTracer(max_depth=2, samples_per_pixel=1)
    
    # Small image for speed
    print("Rendering 100x100 test image...")
    image_array = ray_tracer.render(scene, camera, 100, 100)
    
    # Save test image
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save("test_render.png")
    
    print("✅ Test successful! Check 'test_render.png'")
    print("You should see a red sphere on a dark background.")

if __name__ == "__main__":
    quick_test()