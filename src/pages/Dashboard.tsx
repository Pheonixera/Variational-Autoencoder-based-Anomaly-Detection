
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import NetworkGraph from "@/components/NetworkGraph";
import { AlertTriangle, Check, Gauge, Shield, ShieldAlert } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ScanButton } from "@/components/ScanButton";

const Dashboard = () => {
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Network Overview</h2>
        <div className="flex items-center gap-4">
          <ScanButton />
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="bg-secondary/30 text-secondary-foreground">
              Last scan: 3 minutes ago
            </Badge>
            <Badge variant="outline" className="flex items-center gap-1 bg-primary/20 text-primary">
              <Shield size={14} /> Protected
            </Badge>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-card/50 backdrop-blur border-muted">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Active Devices</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground mt-1">2 new devices connected today</p>
          </CardContent>
        </Card>
        
        <Card className="bg-card/50 backdrop-blur border-muted">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Detected Threats</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-accent">3</div>
            <p className="text-xs text-muted-foreground mt-1">1 critical, 2 moderate</p>
          </CardContent>
        </Card>
        
        <Card className="bg-card/50 backdrop-blur border-muted">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Current Risk Level</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-2xl font-bold flex items-center gap-2">
              Medium <Gauge size={20} className="text-yellow-500" />
            </div>
            <Progress value={45} className="h-2 bg-secondary" />
          </CardContent>
        </Card>
        
        <Card className="bg-card/50 backdrop-blur border-muted">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">System Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Firewall</span>
                <Badge variant="outline" className="bg-green-500/20 text-green-400">
                  <Check size={12} className="mr-1" /> Active
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">IDS Engine</span>
                <Badge variant="outline" className="bg-green-500/20 text-green-400">
                  <Check size={12} className="mr-1" /> Active
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Updates</span>
                <Badge variant="outline" className="bg-yellow-500/20 text-yellow-400">
                  <AlertTriangle size={12} className="mr-1" /> Available
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      <Card className="bg-card/50 backdrop-blur">
        <CardHeader>
          <CardTitle>Network Topology</CardTitle>
        </CardHeader>
        <CardContent>
          <NetworkGraph />
        </CardContent>
      </Card>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle>Recent Anomalies</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { 
                  id: 1, 
                  device: "Client 3", 
                  type: "Unusual Traffic Pattern", 
                  severity: "critical",
                  time: "12:45 PM" 
                },
                { 
                  id: 2, 
                  device: "Database Server", 
                  type: "Multiple Failed Auth", 
                  severity: "warning",
                  time: "11:23 AM" 
                },
                { 
                  id: 3, 
                  device: "IoT Device 2", 
                  type: "Suspicious Connection", 
                  severity: "warning",
                  time: "10:17 AM" 
                }
              ].map(anomaly => (
                <div key={anomaly.id} className="flex items-center justify-between p-2 rounded-md bg-secondary/30">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      {anomaly.severity === "critical" ? (
                        <ShieldAlert size={16} className="text-accent" />
                      ) : (
                        <AlertTriangle size={16} className="text-yellow-400" />
                      )}
                      {anomaly.device}
                    </div>
                    <div className="text-sm text-muted-foreground">{anomaly.type}</div>
                  </div>
                  <div>
                    <Badge 
                      variant={anomaly.severity === "critical" ? "destructive" : "outline"}
                      className={`${anomaly.severity === "warning" ? "bg-yellow-500/20 text-yellow-400" : ""}`}
                    >
                      {anomaly.severity}
                    </Badge>
                    <div className="text-xs text-muted-foreground mt-1">{anomaly.time}</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle>Traffic Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[200px] flex items-center justify-center">
              <span className="text-muted-foreground">Traffic visualization will appear here</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
