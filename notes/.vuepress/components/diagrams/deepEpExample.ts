export type DeepEPMode = "throughput" | "latency"

export type DeepEPRank = {
  node: number
  rank: number
  gpu: number
  experts: readonly [number, number]
}

export type DeepEPBufferSlot = {
  id: string
  source: number
  expert: 12 | 13
  label: string
  focus?: true
}

export const deepEpRanks: readonly DeepEPRank[] = Array.from({ length: 8 }, (_, rank) => ({
  node: rank < 4 ? 0 : 1,
  rank,
  gpu: rank % 4,
  experts: [rank * 2, rank * 2 + 1] as const,
}))

export const deepEpAssignments = [
  { label: "r1:E3", expert: 3, rank: 1, slot: 0, weight: 0.35, remote: false },
  { label: "r1:E13", expert: 13, rank: 6, slot: 1, weight: 0.65, remote: true },
] as const

export const r6Counts = [2, 3, 1, 0, 1, 0, 2, 1] as const
export const r6Offsets = [0, 2, 5, 6, 6, 7, 7, 9, 10] as const

export const compactSlots: readonly DeepEPBufferSlot[] = [
  { id: "s0", source: 0, expert: 12, label: "r0:E12" },
  { id: "s1", source: 0, expert: 13, label: "r0:E13" },
  { id: "s2", source: 1, expert: 12, label: "r1:E12" },
  { id: "focus", source: 1, expert: 13, label: "r1:E13", focus: true },
  { id: "s4", source: 1, expert: 13, label: "r1:E13" },
  { id: "s5", source: 2, expert: 12, label: "r2:E12" },
  { id: "s6", source: 4, expert: 13, label: "r4:E13" },
  { id: "s7", source: 6, expert: 12, label: "r6:E12" },
  { id: "s8", source: 6, expert: 13, label: "r6:E13" },
  { id: "s9", source: 7, expert: 12, label: "r7:E12" },
]

export const expertSlotIndices = {
  12: [0, 2, 5, 7, 9],
  13: [1, 3, 4, 6, 8],
} as const

export const pathSteps = ["Router", "接收空间", "Dispatch", "接收布局", "Expert Compute", "Combine"] as const

export const defaultDedupExperts = [8, 9, 10, 12] as const

export const rankForExpert = (expert: number): number => Math.floor(expert / 2)

export const nodeForRank = (rank: number): number => Math.floor(rank / 4)
