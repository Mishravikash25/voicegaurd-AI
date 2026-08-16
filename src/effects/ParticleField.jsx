import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function Points() {
  const mesh = useRef();
  const n = 1500;
  
  const [coords, sizes] = useMemo(() => {
    const initialCoords = [];
    const initialSizes = [];
    for (let i = 0; i < n; i++) {
        // Spread particles in a 3D sphere/box
        initialCoords.push((Math.random() - 0.5) * 12);
        initialCoords.push((Math.random() - 0.5) * 12);
        initialCoords.push((Math.random() - 0.5) * 12);
        initialSizes.push(Math.random());
    }
    return [new Float32Array(initialCoords), new Float32Array(initialSizes)];
  }, [n]);

  useFrame((state) => {
    const time = state.clock.getElapsedTime();
    if (mesh.current) {
      mesh.current.rotation.y = time * 0.04;
      mesh.current.rotation.x = time * 0.02;
    }
  });

  return (
    <points ref={mesh}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={n}
          array={coords}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.015}
        color="#6366F1"
        transparent
        opacity={0.3}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

const ParticleField = () => {
  return (
    <div className="absolute inset-0 z-0 opacity-40 mix-blend-screen pointer-events-none">
      <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
        <Points />
      </Canvas>
    </div>
  );
};

export default ParticleField;
