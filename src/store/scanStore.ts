
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { v4 as uuidv4 } from 'uuid';
import { ScanResult } from '@/types/scanTypes';

interface ScanState {
  results: ScanResult[];
  addResult: (result: Omit<ScanResult, 'id'>) => void;
  clearResults: () => void;
}

// Sample outputs provided by the user
const sampleOutputs = [
  {
    timestamp: '2025-04-17T07:48:35.895429Z',
    src_ip: '192.168.1.100',
    dst_ip: '10.0.0.5',
    anomaly_type: 'Network Traffic Anomaly',
    severity: 'Medium',
    deviation_score: 19.779218673706055,
    description: 'Detected abnormal network pattern with reconstruction error 0.1978',
    action_taken: 'Logged for review',
    protocol: 'tcp',
    reconstruction_error: 0.1978
  },
  {
    timestamp: '2025-04-17T07:48:35.895429Z',
    src_ip: '192.168.1.100',
    dst_ip: '10.0.0.5',
    anomaly_type: 'Network Traffic Anomaly',
    severity: 'Medium',
    deviation_score: 16.342512130737305,
    description: 'Detected abnormal network pattern with reconstruction error 0.1634',
    action_taken: 'Logged for review',
    protocol: 'tcp',
    reconstruction_error: 0.1634
  },
  {
    timestamp: '2025-04-17T07:48:35.895429Z',
    src_ip: '192.168.1.100',
    dst_ip: '10.0.0.5',
    anomaly_type: 'Network Traffic Anomaly',
    severity: 'Medium',
    deviation_score: 13.921464920043945,
    description: 'Detected abnormal network pattern with reconstruction error 0.1392',
    action_taken: 'Logged for review',
    protocol: 'tcp',
    reconstruction_error: 0.1392
  },
  {
    timestamp: '2025-04-17T07:48:35.895429Z',
    src_ip: '192.168.1.100',
    dst_ip: '10.0.0.5',
    anomaly_type: 'Network Traffic Anomaly',
    severity: 'Medium',
    deviation_score: 14.2427339553833,
    description: 'Detected abnormal network pattern with reconstruction error 0.1424',
    action_taken: 'Logged for review',
    protocol: 'tcp',
    reconstruction_error: 0.1424
  },
  {
    timestamp: '2025-04-17T07:48:35.895429Z',
    src_ip: '192.168.1.100',
    dst_ip: '10.0.0.5',
    anomaly_type: 'Network Traffic Anomaly',
    severity: 'Medium',
    deviation_score: 16.739728927612305,
    description: 'Detected abnormal network pattern with reconstruction error 0.1674',
    action_taken: 'Logged for review',
    protocol: 'tcp',
    reconstruction_error: 0.1674
  }
];

// Mock scan result generator since we can't integrate with the user's model
const generateMockScanResult = (): Omit<ScanResult, 'id'> => {
  // Either select one of the sample outputs or generate a random one
  if (Math.random() > 0.5) {
    // Use one of the sample outputs
    return { ...sampleOutputs[Math.floor(Math.random() * sampleOutputs.length)] };
  }

  // Original random generation logic
  const anomalyTypes = ['Unusual Traffic Pattern', 'Port Scan', 'DNS Anomaly', 'Data Exfiltration', 'Multiple Login Failures'];
  const severityLevels = ['critical', 'warning', 'low', 'info'];
  const protocols = ['TCP', 'UDP', 'HTTP', 'DNS', 'SMTP'];
  const actions = ['Blocked', 'Alerted', 'Monitored', 'Isolated'];
  
  const randomIP = () => 
    `${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`;

  const reconstruction_error = parseFloat((Math.random() * 0.3).toFixed(4));
  const isAnomaly = reconstruction_error > 0.13;
  
  return {
    timestamp: new Date().toISOString(),
    src_ip: randomIP(),
    dst_ip: randomIP(),
    anomaly_type: isAnomaly ? 'Anomaly Detected' : 'Normal Traffic',
    severity: isAnomaly ? severityLevels[Math.floor(Math.random() * 2)] : 'info',
    deviation_score: parseFloat((Math.random() * 20).toFixed(2)),
    description: isAnomaly 
      ? `Detected anomaly with reconstruction error ${reconstruction_error}`
      : `Normal traffic pattern with reconstruction error ${reconstruction_error}`,
    action_taken: isAnomaly ? actions[Math.floor(Math.random() * actions.length)] : 'None',
    protocol: protocols[Math.floor(Math.random() * protocols.length)].toLowerCase(),
    reconstruction_error: reconstruction_error
  };
};

export const useScanStore = create<ScanState>()(
  persist(
    (set) => ({
      results: [],
      addResult: (result) => set((state) => ({
        results: [...state.results, { ...result, id: uuidv4() }]
      })),
      clearResults: () => set({ results: [] }),
    }),
    {
      name: 'scan-results',
    }
  )
);

// Function to initiate scan and get results
export const performNetworkScan = async () => {
  // In a real implementation, this would connect to the user's model
  // Using a longer delay of 8 seconds as requested
  await new Promise(resolve => setTimeout(resolve, 8000));
  
  const mockResult = generateMockScanResult();
  useScanStore.getState().addResult(mockResult);
  return mockResult;
};
