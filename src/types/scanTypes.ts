
export interface ScanResult {
  id: string;
  timestamp: string;
  src_ip: string;
  dst_ip: string;
  anomaly_type: string;
  severity: string;
  deviation_score: number;
  description: string;
  action_taken: string;
  protocol: string;
  reconstruction_error?: number;
}
