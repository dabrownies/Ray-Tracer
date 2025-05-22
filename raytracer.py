import numpy as np
import math
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image
import random

# basic 3d vector math - handles all the coordinate stuff
class Vec3:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x, self.y, self.z = x, y, z
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
        else:  # element-wise for colors
            return Vec3(self.x * scalar.x, self.y * scalar.y, self.z * scalar.z)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return self / length
        return Vec3(0, 0, 0)
    
    def reflect(self, normal):
        # mirror reflection formula
        return self - 2 * self.dot(normal) * normal
    
    def refract(self, normal, eta_ratio):
        # snell's law for glass/water bending
        cos_i = -self.dot(normal)
        sin_t2 = eta_ratio * eta_ratio * (1.0 - cos_i * cos_i)
        
        if sin_t2 >= 1.0:  # total internal reflection
            return None
        
        cos_t = math.sqrt(1.0 - sin_t2)
        return eta_ratio * self + (eta_ratio * cos_i - cos_t) * normal
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])


# a ray is just a starting point + direction
@dataclass
class Ray:
    origin: Vec3
    direction: Vec3
    
    def at(self, t: float) -> Vec3:
        return self.origin + t * self.direction


# material properties - how light bounces off stuff
class Material:
    def __init__(self, color: Vec3, ambient: float = 0.1, diffuse: float = 0.7, 
                 specular: float = 0.2, shininess: float = 32, reflectivity: float = 0.0,
                 transparency: float = 0.0, ior: float = 1.0):
        self.color = color
        self.ambient = ambient          # base brightness
        self.diffuse = diffuse          # matte surface scattering
        self.specular = specular        # shiny highlights
        self.shininess = shininess      # how tight the highlights are
        self.reflectivity = reflectivity # mirror amount
        self.transparency = transparency # glass amount
        self.ior = ior                  # index of refraction


# textures for patterns on surfaces
class Texture:
    def get_color(self, u: float, v: float) -> Vec3:
        return Vec3(1, 1, 1)

class CheckerTexture(Texture):
    def __init__(self, color1: Vec3, color2: Vec3, scale: float = 10.0):
        self.color1 = color1
        self.color2 = color2
        self.scale = scale
    
    def get_color(self, u: float, v: float) -> Vec3:
        # checkerboard pattern based on texture coords
        pattern = (int(u * self.scale) + int(v * self.scale)) % 2
        return self.color1 if pattern == 0 else self.color2


# info about where a ray hits something
@dataclass
class HitRecord:
    point: Vec3
    normal: Vec3
    t: float        # distance along ray
    material: Material
    u: float = 0.0  # texture coordinates
    v: float = 0.0
    texture: Optional[Texture] = None


# sphere object
class Sphere:
    def __init__(self, center: Vec3, radius: float, material: Material, texture: Optional[Texture] = None):
        self.center = center
        self.radius = radius
        self.material = material
        self.texture = texture
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        # solve quadratic equation for ray-sphere intersection
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        # find closest intersection point
        sqrt_discriminant = math.sqrt(discriminant)
        root = (-b - sqrt_discriminant) / (2.0 * a)
        
        if root < t_min or root > t_max:
            root = (-b + sqrt_discriminant) / (2.0 * a)
            if root < t_min or root > t_max:
                return None
        
        point = ray.at(root)
        normal = (point - self.center).normalize()
        
        # spherical texture coordinates
        theta = math.acos(-normal.y)
        phi = math.atan2(-normal.z, normal.x) + math.pi
        u = phi / (2 * math.pi)
        v = theta / math.pi
        
        return HitRecord(point, normal, root, self.material, u, v, self.texture)


# flat plane object
class Plane:
    def __init__(self, point: Vec3, normal: Vec3, material: Material, texture: Optional[Texture] = None):
        self.point = point
        self.normal = normal.normalize()
        self.material = material
        self.texture = texture
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        # check if ray is parallel to plane
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 1e-6:
            return None
        
        t = (self.point - ray.origin).dot(self.normal) / denom
        if t < t_min or t > t_max:
            return None
        
        point = ray.at(t)
        
        # simple planar texture mapping
        u = point.x % 1.0
        v = point.z % 1.0
        
        return HitRecord(point, self.normal, t, self.material, u, v, self.texture)


# light source
class Light:
    def __init__(self, position: Vec3, color: Vec3, intensity: float = 1.0):
        self.position = position
        self.color = color
        self.intensity = intensity


# holds all the objects and lights
class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.background_color = Vec3(0.1, 0.1, 0.2)
        self.ambient_light = Vec3(0.1, 0.1, 0.1)
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light: Light):
        self.lights.append(light)
    
    def hit(self, ray: Ray, t_min: float = 0.001, t_max: float = float('inf')) -> Optional[HitRecord]:
        # find closest object the ray hits
        closest_hit = None
        closest_t = t_max
        
        for obj in self.objects:
            hit = obj.hit(ray, t_min, closest_t)
            if hit and hit.t < closest_t:
                closest_t = hit.t
                closest_hit = hit
        
        return closest_hit


# virtual camera that shoots rays
class Camera:
    def __init__(self, position: Vec3, target: Vec3, up: Vec3, fov: float, aspect_ratio: float):
        self.position = position
        self.target = target
        self.up = up
        self.fov = math.radians(fov)
        self.aspect_ratio = aspect_ratio
        
        # build camera coordinate system
        self.forward = (target - position).normalize()
        self.right = self.forward.cross(up).normalize()
        self.up_vec = self.right.cross(self.forward).normalize()
        
        # calculate how big the view plane is
        self.view_height = 2.0 * math.tan(self.fov / 2.0)
        self.view_width = self.view_height * aspect_ratio
    
    def get_ray(self, u: float, v: float) -> Ray:
        # convert pixel coords to world space ray
        u = u * 2.0 - 1.0
        v = v * 2.0 - 1.0
        
        direction = (
            self.forward +
            u * (self.view_width / 2.0) * self.right +
            v * (self.view_height / 2.0) * self.up_vec
        ).normalize()
        
        return Ray(self.position, direction)


# the main ray tracing engine
class RayTracer:
    def __init__(self, max_depth: int = 5, samples_per_pixel: int = 1):
        self.max_depth = max_depth
        self.samples_per_pixel = samples_per_pixel
    
    def trace_ray(self, ray: Ray, scene: Scene, depth: int = 0) -> Vec3:
        # stop bouncing after max depth
        if depth >= self.max_depth:
            return Vec3(0, 0, 0)
        
        hit = scene.hit(ray)
        if not hit:
            return scene.background_color
        
        # get base color from material or texture
        material_color = hit.material.color
        if hit.texture:
            material_color = hit.texture.get_color(hit.u, hit.v)
        
        # calculate how light hits this point
        color = self.calculate_lighting(hit, scene, ray.direction, material_color)
        
        # add mirror reflections
        if hit.material.reflectivity > 0:
            reflect_dir = ray.direction.reflect(hit.normal)
            reflect_ray = Ray(hit.point, reflect_dir)
            reflect_color = self.trace_ray(reflect_ray, scene, depth + 1)
            color = color * (1 - hit.material.reflectivity) + reflect_color * hit.material.reflectivity
        
        # add glass refraction
        if hit.material.transparency > 0:
            refract_color = Vec3(0, 0, 0)
            
            # figure out if ray is entering or leaving glass
            cos_i = -ray.direction.dot(hit.normal)
            if cos_i < 0:  # inside object
                cos_i = -cos_i
                normal = -1 * hit.normal
                eta_ratio = hit.material.ior
            else:  # outside object
                normal = hit.normal
                eta_ratio = 1.0 / hit.material.ior
            
            refracted_dir = ray.direction.refract(normal, eta_ratio)
            if refracted_dir:  # no total internal reflection
                refract_ray = Ray(hit.point, refracted_dir)
                refract_color = self.trace_ray(refract_ray, scene, depth + 1)
            
            # fresnel effect - glass gets more reflective at shallow angles
            fresnel = self.fresnel_schlick(cos_i, hit.material.ior)
            color = color * (1 - hit.material.transparency * (1 - fresnel)) + refract_color * hit.material.transparency * (1 - fresnel)
        
        return color
    
    def fresnel_schlick(self, cos_theta: float, ior: float) -> float:
        # approximation for how much light reflects vs refracts
        r0 = ((1 - ior) / (1 + ior)) ** 2
        return r0 + (1 - r0) * ((1 - cos_theta) ** 5)
    
    def calculate_lighting(self, hit: HitRecord, scene: Scene, view_dir: Vec3, material_color: Vec3) -> Vec3:
        # start with ambient light
        color = scene.ambient_light * material_color * hit.material.ambient
        
        for light in scene.lights:
            light_dir = (light.position - hit.point).normalize()
            light_distance = (light.position - hit.point).length()
            
            # check if this point is in shadow
            shadow_ray = Ray(hit.point, light_dir)
            shadow_hit = scene.hit(shadow_ray, 0.001, light_distance - 0.001)
            
            if not shadow_hit:  # not in shadow
                # diffuse shading - how much surface faces the light
                diffuse_intensity = max(0, hit.normal.dot(light_dir))
                diffuse = material_color * light.color * hit.material.diffuse * diffuse_intensity * light.intensity
                
                # specular highlights - shiny reflections
                half_vector = (light_dir - view_dir).normalize()
                specular_intensity = max(0, hit.normal.dot(half_vector)) ** hit.material.shininess
                specular = light.color * hit.material.specular * specular_intensity * light.intensity
                
                # light gets dimmer with distance
                attenuation = 1.0 / (1.0 + 0.1 * light_distance + 0.01 * light_distance * light_distance)
                
                color = color + (diffuse + specular) * attenuation
        
        return color
    
    def render(self, scene: Scene, camera: Camera, width: int, height: int) -> np.ndarray:
        image = np.zeros((height, width, 3))
        
        print(f"rendering {width}x{height} image with {self.samples_per_pixel} samples per pixel...")
        
        for y in range(height):
            if y % (height // 10) == 0:
                print(f"progress: {y/height*100:.1f}%")
            
            for x in range(width):
                color = Vec3(0, 0, 0)
                
                # anti-aliasing - shoot multiple rays per pixel
                for _ in range(self.samples_per_pixel):
                    # add tiny random offset to smooth edges
                    u = (x + random.random()) / width
                    v = 1.0 - (y + random.random()) / height  # flip y
                    
                    ray = camera.get_ray(u, v)
                    color = color + self.trace_ray(ray, scene)
                
                color = color / self.samples_per_pixel
                
                # gamma correction and clamp to valid range
                color = Vec3(
                    min(1.0, math.sqrt(max(0, color.x))),
                    min(1.0, math.sqrt(max(0, color.y))),
                    min(1.0, math.sqrt(max(0, color.z)))
                )
                
                image[y, x] = [color.x, color.y, color.z]
        
        return image


# set up the demo scene
def create_demo_scene():
    scene = Scene()
    
    # different material types
    red_diffuse = Material(Vec3(0.8, 0.2, 0.2), diffuse=0.8, specular=0.2)
    blue_diffuse = Material(Vec3(0.2, 0.2, 0.8), diffuse=0.7, specular=0.3, shininess=64)
    mirror = Material(Vec3(0.9, 0.9, 0.9), diffuse=0.1, specular=0.1, reflectivity=0.8)
    glass = Material(Vec3(0.95, 0.95, 0.95), diffuse=0.1, specular=0.1, 
                    reflectivity=0.1, transparency=0.9, ior=1.5)
    
    # checkerboard floor
    checker = CheckerTexture(Vec3(0.8, 0.8, 0.8), Vec3(0.2, 0.2, 0.2), scale=4.0)
    floor_material = Material(Vec3(1, 1, 1), diffuse=0.8, specular=0.2)
    
    # add some spheres with different materials
    scene.add_object(Sphere(Vec3(-1, 0, -3), 1, red_diffuse))
    scene.add_object(Sphere(Vec3(1, 0, -3), 1, glass))
    scene.add_object(Sphere(Vec3(0, 2, -4), 0.8, mirror))
    scene.add_object(Sphere(Vec3(-2, -0.5, -2), 0.5, blue_diffuse))
    
    # floor with pattern
    scene.add_object(Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), floor_material, checker))
    
    # couple different colored lights
    scene.add_light(Light(Vec3(2, 4, -1), Vec3(1, 1, 1), intensity=1.0))
    scene.add_light(Light(Vec3(-2, 3, -2), Vec3(0.8, 0.8, 1.0), intensity=0.5))
    
    return scene


def main():
    scene = create_demo_scene()
    
    camera = Camera(
        position=Vec3(0, 1, 1),
        target=Vec3(0, 0, -3),
        up=Vec3(0, 1, 0),
        fov=45,
        aspect_ratio=16/9
    )
    
    # decent quality settings
    ray_tracer = RayTracer(max_depth=6, samples_per_pixel=4)
    
    width, height = 800, 450
    image_array = ray_tracer.render(scene, camera, width, height)
    
    # save as png
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save("raytracer_output.png")
    print("image saved as 'raytracer_output.png'")
    
    print(f"\nrender completed!")
    print(f"resolution: {width}x{height}")
    print(f"anti-aliasing samples: {ray_tracer.samples_per_pixel}")
    print(f"maximum ray depth: {ray_tracer.max_depth}")
    print(f"objects in scene: {len(scene.objects)}")
    print(f"lights in scene: {len(scene.lights)}")


if __name__ == "__main__":
    main()