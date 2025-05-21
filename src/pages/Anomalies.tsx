
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, Clock, Filter, Search, ShieldAlert, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";

const anomaliesData = [
  {
    id: 1,
    deviceName: "Client 3",
    deviceIP: "192.168.1.45",
    type: "Unusual Traffic Pattern",
    description: "Unexpected outbound connection to multiple IPs",
    severity: "critical",
    time: "Today, 12:45 PM",
    status: "active"
  },
  {
    id: 2,
    deviceName: "Database Server",
    deviceIP: "192.168.1.12",
    type: "Multiple Failed Authentication",
    description: "10+ failed login attempts in 5 minutes",
    severity: "warning",
    time: "Today, 11:23 AM",
    status: "active"
  },
  {
    id: 3,
    deviceName: "IoT Device 2",
    deviceIP: "192.168.1.87",
    type: "Suspicious Connection",
    description: "Connection attempt to known malicious IP",
    severity: "warning",
    time: "Today, 10:17 AM",
    status: "active"
  },
  {
    id: 4,
    deviceName: "Web Server",
    deviceIP: "192.168.1.10",
    type: "Port Scan Detected",
    description: "Sequential port scan from external IP",
    severity: "warning",
    time: "Yesterday, 8:30 PM",
    status: "resolved"
  },
  {
    id: 5,
    deviceName: "Router",
    deviceIP: "192.168.1.1",
    type: "DNS Anomaly",
    description: "Unusual DNS query pattern detected",
    severity: "low",
    time: "Yesterday, 5:12 PM",
    status: "resolved"
  },
  {
    id: 6,
    deviceName: "Client 2",
    deviceIP: "192.168.1.35",
    type: "Data Exfiltration Attempt",
    description: "Unusual large outbound data transfer",
    severity: "critical",
    time: "2 Days ago, 3:45 PM",
    status: "resolved"
  }
];

const Anomalies = () => {
  const [filter, setFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredAnomalies = anomaliesData.filter(anomaly => {
    // Filter by status
    if (filter === "active" && anomaly.status !== "active") return false;
    if (filter === "resolved" && anomaly.status !== "resolved") return false;
    
    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        anomaly.deviceName.toLowerCase().includes(query) ||
        anomaly.deviceIP.toLowerCase().includes(query) ||
        anomaly.type.toLowerCase().includes(query) ||
        anomaly.description.toLowerCase().includes(query)
      );
    }
    
    return true;
  });
  
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "critical":
        return <ShieldAlert size={16} className="text-accent" />;
      case "warning":
        return <AlertTriangle size={16} className="text-yellow-400" />;
      default:
        return <ShieldCheck size={16} className="text-green-400" />;
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case "critical":
        return <Badge variant="destructive">Critical</Badge>;
      case "warning":
        return <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-800">Warning</Badge>;
      default:
        return <Badge className="bg-green-500/20 text-green-400 border-green-800">Low</Badge>;
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Anomalies</h2>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input 
            placeholder="Search anomalies..." 
            className="pl-8" 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        
        <div className="flex gap-2 w-full sm:w-auto">
          <div className="w-full sm:w-40">
            <Select value={filter} onValueChange={setFilter}>
              <SelectTrigger>
                <div className="flex items-center gap-2">
                  <Filter size={14} />
                  <SelectValue placeholder="Filter" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Anomalies</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="resolved">Resolved</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <Button variant="secondary">
            Export
          </Button>
        </div>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Detected Anomalies</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredAnomalies.length > 0 ? (
              filteredAnomalies.map(anomaly => (
                <div 
                  key={anomaly.id} 
                  className={`p-4 rounded-md border ${
                    anomaly.status === 'resolved' 
                      ? 'bg-secondary/10 border-muted' 
                      : 'bg-secondary/30 border-secondary'
                  }`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        {getSeverityIcon(anomaly.severity)}
                        <span className="font-medium">{anomaly.type}</span>
                        {getSeverityBadge(anomaly.severity)}
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">{anomaly.description}</p>
                      <div className="flex items-center gap-4 text-xs">
                        <span className="text-muted-foreground">{anomaly.deviceName} ({anomaly.deviceIP})</span>
                        <span className="flex items-center gap-1 text-muted-foreground">
                          <Clock size={12} /> {anomaly.time}
                        </span>
                      </div>
                    </div>
                    
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm">Details</Button>
                      {anomaly.status === 'active' && (
                        <Button size="sm">Resolve</Button>
                      )}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center p-8 text-muted-foreground">
                No anomalies match your criteria
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Anomalies;
