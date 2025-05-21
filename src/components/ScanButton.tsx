
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Radar, Loader2 } from 'lucide-react';
import { performNetworkScan } from '@/store/scanStore';
import { useToast } from '@/hooks/use-toast';
import { useNavigate } from 'react-router-dom';

export function ScanButton() {
  const [isScanning, setIsScanning] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  const handleScan = async () => {
    setIsScanning(true);
    try {
      toast({
        title: "Scan Started",
        description: "Analyzing network traffic patterns...",
      });
      
      const result = await performNetworkScan();
      
      // Determine if it's an anomaly based on reconstruction error
      const isAnomaly = result.reconstruction_error !== undefined && result.reconstruction_error > 0.13;
      
      toast({
        title: "Scan Complete",
        description: isAnomaly 
          ? `Detected anomaly with reconstruction error ${result.reconstruction_error}` 
          : "No significant anomalies detected",
        variant: isAnomaly ? "destructive" : "default",
      });

      // Navigate to history page after scan completes
      navigate('/history');
      
    } catch (error) {
      toast({
        title: "Scan Failed",
        description: "Unable to complete network scan",
        variant: "destructive",
      });
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <Button 
      onClick={handleScan}
      disabled={isScanning}
      className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transition-all"
    >
      {isScanning ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Scanning...
        </>
      ) : (
        <>
          <Radar className="mr-2 h-4 w-4" />
          Scan Network
        </>
      )}
    </Button>
  );
}
