export interface BandActivity {
  band: number;
  freqMhz: number;
  activityLevel: number;
  detectedCount?: number;
}

export interface FOMItem {
  id: string;
  num?: string;
  name: string;
  symbol: string;
  value: number;
  formatted: string;
  unit?: string;
  passCriteria: string;
  formula: string;
  status: string;
  desc?: string;
  description?: string;
}

export interface SystemStatus {
  mode: string;
  backendConnected: boolean;
  environment: string;
  schedulerName: string;
  episode: number;
  totalEpisodes: number;
  step: number;
  totalSteps: number;
  runStatus: string;
  errorMessage: string | null;
  datasetLoaded: boolean;
  datasetSize: number | null;
  /** Number of distinct TSRD configs loaded (config_0 + folder-3 configs) */
  configsLoaded?: number;
  /** Active config ID being used for the current/last simulation */
  activeConfig?: string;
}

/** Metadata for a single TSRD scenario configuration file */
export interface TSRDConfig {
  configId: string;
  /** Original config ID before any prefix (e.g. "config_1" for "stare_config_1") */
  rawConfigId?: string;
  filename: string;
  /**
   * Receiver operating mode:
   * - "scan"  — receiver sweeps bands (Folder 3, dwell_centres_mhz populated)
   * - "stare" — receiver fixed at one frequency (Folder 4, dwell_centres_mhz empty)
   */
  dataMode?: "scan" | "stare";
  pulseCount: number;
  txCount: number;
  uniqueEmitters: number;
  freqMinMhz: number;
  freqMaxMhz: number;
  freqRangeMhz: [number, number];
  rxPositionKm: [number, number];
  dwellCentresMhz: number[];
  dwellTimesS: number[];
  /** Fraction of time slots each of the 36 bands is active [0..1] */
  bandActivity: number[];
}

/** Response from /api/dataset/configs */
export interface DatasetConfigsResponse {
  configs: TSRDConfig[];
  totalConfigs: number;
  totalPulses: number;
  totalEmitters: number;
  /** Number of scan-mode configs (Folder 3 + config_0) */
  scanConfigCount?: number;
  /** Number of stare-mode configs (Folder 4) */
  stareConfigCount?: number;
  datasetLoaded: boolean;
  episodesAvailable: number | null;
  scanStats: Array<{
    dir: string;
    label: string;
    count: number;
    by_file?: Record<string, number>;
  }> | null;
}

/** A single emitter entry from /api/emitters or /api/dataset/config/{id} */
export interface TSRDEmitter {
  id: string;
  configId: string;
  transmitterId?: number;
  index?: number;
  type: string;
  role?: string;
  band: number;
  freq: number;
  freq_mhz?: number;
  freqs_mhz?: number[];
  pri: string;
  pris_us?: number[];
  pw: string;
  pws_us?: number[];
  power: string;
  pos_km?: number[];
  pulses?: number;
  active: boolean;
  agility: number;
  detected: boolean;
}
