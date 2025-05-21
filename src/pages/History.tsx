
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { 
  AlertTriangle, 
  Calendar, 
  Clock, 
  Download, 
  Filter,
  Search, 
  Shield, 
  ShieldAlert, 
  ShieldCheck,
  Radar
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useScanStore } from "@/store/scanStore";

// Mock history data
const historyEvents = [
  {
    id: 1,
    type: "alert",
    title: "Critical Threat Detected",
    description: "Data exfiltration attempt blocked from Client 2",
    time: "2023-04-14T15:45:00",
    severity: "critical",
  },
  {
    id: 2,
    type: "alert",
    title: "Multiple Login Failures",
    description: "10+ failed login attempts to Database Server",
    time: "2023-04-14T11:23:00",
    severity: "warning",
  },
  {
    id: 3,
    type: "system",
    title: "IDS Engine Updated",
    description: "System updated to version 3.1.4",
    time: "2023-04-14T10:00:00",
    severity: "info",
  },
  {
    id: 4,
    type: "alert",
    title: "Suspicious Connection",
    description: "Connection attempt to known malicious IP from IoT Device 2",
    time: "2023-04-14T08:17:00",
    severity: "warning",
  },
  {
    id: 5,
    type: "alert",
    title: "Port Scan Detected",
    description: "Sequential port scan from external IP to Web Server",
    time: "2023-04-13T20:30:00",
    severity: "warning",
  },
  {
    id: 6,
    type: "system",
    title: "Backup Completed",
    description: "System configuration backup completed successfully",
    time: "2023-04-13T18:00:00",
    severity: "info",
  },
  {
    id: 7,
    type: "user",
    title: "User Login",
    description: "Administrator logged in from 192.168.1.100",
    time: "2023-04-13T16:45:00",
    severity: "info",
  },
  {
    id: 8,
    type: "alert",
    title: "DNS Anomaly",
    description: "Unusual DNS query pattern detected from Router",
    time: "2023-04-13T17:12:00",
    severity: "low",
  },
  {
    id: 9,
    type: "user",
    title: "Rule Modified",
    description: "Firewall rule #12 modified by Administrator",
    time: "2023-04-13T15:30:00",
    severity: "info",
  },
  {
    id: 10,
    type: "alert",
    title: "Critical Threat Detected",
    description: "Unusual large outbound data transfer from Client 2",
    time: "2023-04-12T15:45:00",
    severity: "critical",
  },
];

const History = () => {
  const [typeFilter, setTypeFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const { results: scanResults } = useScanStore();
  
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };
  
  const filteredEvents = historyEvents.filter(event => {
    // Filter by type
    if (typeFilter !== "all" && event.type !== typeFilter) return false;
    
    // Filter by severity
    if (severityFilter !== "all" && event.severity !== severityFilter) return false;
    
    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        event.title.toLowerCase().includes(query) ||
        event.description.toLowerCase().includes(query)
      );
    }
    
    return true;
  });
  
  const getEventIcon = (event: any) => {
    if (event.type === "system") {
      return <Shield size={18} className="text-primary" />;
    }
    
    if (event.type === "user") {
      return <Shield size={18} className="text-blue-400" />;
    }
    
    // For alerts, use severity
    switch (event.severity) {
      case "critical":
        return <ShieldAlert size={18} className="text-accent" />;
      case "warning":
        return <AlertTriangle size={18} className="text-yellow-400" />;
      case "low":
        return <AlertTriangle size={18} className="text-blue-400" />;
      default:
        return <ShieldCheck size={18} className="text-green-400" />;
    }
  };
  
  const getSeverityClass = (severity: string) => {
    switch (severity) {
      case "critical":
        return "text-accent border-accent/20 bg-accent/10";
      case "warning":
        return "text-yellow-400 border-yellow-400/20 bg-yellow-400/10";
      case "low":
        return "text-blue-400 border-blue-400/20 bg-blue-400/10";
      default:
        return "text-green-400 border-green-400/20 bg-green-400/10";
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Event History</h2>
        <Button variant="outline" size="sm">
          <Calendar className="mr-2 h-4 w-4" />
          Date Range
        </Button>
      </div>
      
      <div className="flex flex-col md:flex-row gap-3 items-start md:items-center justify-between">
        <div className="relative w-full md:w-64">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input 
            placeholder="Search events..." 
            className="pl-8" 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        
        <div className="flex gap-2 w-full md:w-auto">
          <div className="w-full md:w-40">
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger>
                <div className="flex items-center gap-2">
                  <Filter size={14} />
                  <SelectValue placeholder="Event Type" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="alert">Alerts</SelectItem>
                <SelectItem value="system">System</SelectItem>
                <SelectItem value="user">User</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="w-full md:w-40">
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger>
                <div className="flex items-center gap-2">
                  <AlertTriangle size={14} />
                  <SelectValue placeholder="Severity" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="info">Info</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <Button variant="outline" size="icon">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      {scanResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Radar className="h-5 w-5 text-primary" />
              Scan Results
            </CardTitle>
            <CardDescription>
              Results from network scans performed by the detection model
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Source IP</TableHead>
                    <TableHead>Destination IP</TableHead>
                    <TableHead>Protocol</TableHead>
                    <TableHead>Anomaly Type</TableHead>
                    <TableHead>Reconstruction Error</TableHead>
                    <TableHead>Severity</TableHead>
                    <TableHead>Action Taken</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {scanResults.map((result) => {
                    const isAnomaly = result.reconstruction_error !== undefined && result.reconstruction_error > 0.13;
                    return (
                      <TableRow key={result.id} className={isAnomaly ? "bg-destructive/10" : ""}>
                        <TableCell>{formatDate(result.timestamp)}</TableCell>
                        <TableCell>{result.src_ip}</TableCell>
                        <TableCell>{result.dst_ip}</TableCell>
                        <TableCell>{result.protocol}</TableCell>
                        <TableCell>{result.anomaly_type}</TableCell>
                        <TableCell>
                          {result.reconstruction_error !== undefined && (
                            <span className={`font-mono ${isAnomaly ? 'text-destructive font-medium' : ''}`}>
                              {result.reconstruction_error.toFixed(4)}
                              {isAnomaly && ' ⚠️'}
                            </span>
                          )}
                        </TableCell>
                        <TableCell>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            result.severity === 'critical' ? 'bg-destructive/20 text-destructive' :
                            result.severity === 'warning' || result.severity === 'Medium' ? 'bg-yellow-500/20 text-yellow-400' :
                            result.severity === 'low' ? 'bg-blue-500/20 text-blue-400' :
                            'bg-green-500/20 text-green-400'
                          }`}>
                            {result.severity}
                          </span>
                        </TableCell>
                        <TableCell>{result.action_taken}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
      
      <Card>
        <CardHeader>
          <CardTitle>System Events</CardTitle>
          <CardDescription>
            Full history of system events, alerts and user activities
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredEvents.length > 0 ? (
              filteredEvents.map(event => (
                <div 
                  key={event.id} 
                  className="p-3 border border-secondary/50 rounded-md bg-secondary/20 hover:bg-secondary/30 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="h-8 w-8 rounded-full bg-background flex items-center justify-center">
                        {getEventIcon(event)}
                      </div>
                      <div>
                        <h4 className="font-medium">{event.title}</h4>
                        <p className="text-sm text-muted-foreground">{event.description}</p>
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      {event.severity !== 'info' && (
                        <span className={`text-xs px-2 py-1 border rounded ${getSeverityClass(event.severity)}`}>
                          {event.severity}
                        </span>
                      )}
                      <span className="text-xs flex items-center gap-1 text-muted-foreground">
                        <Clock size={12} />
                        {formatDate(event.time)}
                      </span>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center p-8 text-muted-foreground">
                No events match your filter criteria
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default History;
