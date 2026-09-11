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
}
