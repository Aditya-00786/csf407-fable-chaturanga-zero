/** Development aid (?showcase, or ?showcase&kind=3 for one piece up close): piece models lined up, for judging silhouettes. */

import { Environment, Lightformer, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { PIECE_ORDER } from '../lib/pieces'
import { PieceModel } from './PieceModel'

export function Showcase() {
  const only = Number(new URLSearchParams(location.search).get('kind')) || null
  const kinds = only ? PIECE_ORDER.filter((k) => k === only) : PIECE_ORDER
  return (
    <Canvas shadows="percentage" camera={{ fov: 28, position: only ? [1.2, 1.4, 3] : [0, 2.2, 8] }}>
      <OrbitControls target={only ? [0, 0.55, -0.6] : [0, 0.6, -0.6]} />
      <hemisphereLight args={['#ffe0b8', '#2b140c', 0.5]} />
      <directionalLight position={[3, 6, 5]} intensity={2.5} color="#ffd29a" castShadow />
      <Environment resolution={256} environmentIntensity={0.6}>
        <Lightformer form="rect" intensity={2.2} color="#ffe6c4" position={[0, 6, -4]} scale={[12, 4, 1]} />
        <Lightformer form="rect" intensity={1.2} color="#fff2e0" position={[-6, 3, 2]} rotation={[0, Math.PI / 2, 0]} scale={[8, 2, 1]} />
      </Environment>
      {kinds.map((kind, i) => (
        <group key={kind} position={[(i - (kinds.length - 1) / 2) * 1.05, 0, 0]}>
          <PieceModel kind={kind} white />
          <group position={[0, 0, -1.2]} scale={[-1, 1, 1]}>
            <PieceModel kind={kind} white={false} />
          </group>
        </group>
      ))}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#b5804a" />
      </mesh>
    </Canvas>
  )
}
