
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from "@/components/ui/tabs";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { AlertCircle, Bell, Check, Cloud, Database, Network, Shield, User } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

const Settings = () => {
  const [scanFrequency, setScanFrequency] = useState("hourly");
  const [alertThreshold, setAlertThreshold] = useState("medium");
  const [autoUpdates, setAutoUpdates] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [smsNotifications, setSmsNotifications] = useState(false);
  const [dashboardNotifications, setDashboardNotifications] = useState(true);
  
  const saveSettings = () => {
    toast.success("Settings saved successfully", {
      description: "Your configuration changes have been applied.",
      icon: <Check className="h-4 w-4" />,
    });
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Settings</h2>
      </div>
      
      <Tabs defaultValue="general" className="space-y-4">
        <TabsList className="grid grid-cols-5 sm:w-[600px]">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="scanning">Scanning</TabsTrigger>
          <TabsTrigger value="notifications">Alerts</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="users">Users</TabsTrigger>
        </TabsList>
        
        {/* General Settings */}
        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
              <CardDescription>
                Configure basic system settings and preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-lg font-medium">System</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="system-name">System Name</Label>
                    <Input id="system-name" defaultValue="SENTINEL IDS" />
                  </div>
                  <div className="space-y-2">
                    <Label>Timezone</Label>
                    <Select defaultValue="utc">
                      <SelectTrigger>
                        <SelectValue placeholder="Select timezone" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="utc">UTC (Coordinated Universal Time)</SelectItem>
                        <SelectItem value="est">EST (Eastern Standard Time)</SelectItem>
                        <SelectItem value="cst">CST (Central Standard Time)</SelectItem>
                        <SelectItem value="mst">MST (Mountain Standard Time)</SelectItem>
                        <SelectItem value="pst">PST (Pacific Standard Time)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch
                    id="auto-updates"
                    checked={autoUpdates}
                    onCheckedChange={setAutoUpdates}
                  />
                  <Label htmlFor="auto-updates">Enable automatic updates</Label>
                </div>
                
                <div className="bg-secondary/20 p-4 rounded border border-secondary">
                  <div className="flex items-center gap-2">
                    <Cloud className="text-primary" size={16} />
                    <h4 className="font-medium">Cloud Services</h4>
                  </div>
                  <p className="text-sm text-muted-foreground mt-1 mb-2">
                    Connect to our threat intelligence cloud for enhanced detection capabilities
                  </p>
                  <div className="flex gap-2">
                    <Button variant="secondary" size="sm">Configure</Button>
                    <Badge variant="outline" className="bg-primary/20 text-primary">Connected</Badge>
                  </div>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Logging</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Log Retention Period</Label>
                    <Select defaultValue="30">
                      <SelectTrigger>
                        <SelectValue placeholder="Select period" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="7">7 days</SelectItem>
                        <SelectItem value="14">14 days</SelectItem>
                        <SelectItem value="30">30 days</SelectItem>
                        <SelectItem value="90">90 days</SelectItem>
                        <SelectItem value="180">6 months</SelectItem>
                        <SelectItem value="365">1 year</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Log Level</Label>
                    <Select defaultValue="info">
                      <SelectTrigger>
                        <SelectValue placeholder="Select level" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="error">Error only</SelectItem>
                        <SelectItem value="warn">Warning & Error</SelectItem>
                        <SelectItem value="info">Info & above</SelectItem>
                        <SelectItem value="debug">Debug & above</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="syslog-export" />
                  <Label htmlFor="syslog-export">Export logs to Syslog server</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="db-logging" />
                  <Label htmlFor="db-logging">Store detailed event logs in database</Label>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button onClick={saveSettings}>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Scanning Settings */}
        <TabsContent value="scanning">
          <Card>
            <CardHeader>
              <CardTitle>Scanning Settings</CardTitle>
              <CardDescription>
                Configure how and when the system scans for threats
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Scan Configuration</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Scan Frequency</Label>
                    <Select
                      value={scanFrequency}
                      onValueChange={setScanFrequency}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select frequency" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="realtime">Real-time monitoring</SelectItem>
                        <SelectItem value="hourly">Hourly</SelectItem>
                        <SelectItem value="daily">Daily</SelectItem>
                        <SelectItem value="weekly">Weekly</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Scan Depth</Label>
                    <Select defaultValue="deep">
                      <SelectTrigger>
                        <SelectValue placeholder="Select depth" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="quick">Quick scan (low resource usage)</SelectItem>
                        <SelectItem value="standard">Standard scan (balanced)</SelectItem>
                        <SelectItem value="deep">Deep scan (thorough)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label>Alert Threshold</Label>
                  <Select
                    value={alertThreshold}
                    onValueChange={setAlertThreshold}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select threshold" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low (more alerts)</SelectItem>
                      <SelectItem value="medium">Medium (balanced)</SelectItem>
                      <SelectItem value="high">High (fewer alerts)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Detection Methods</h3>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Switch defaultChecked id="signature-based" />
                      <Label htmlFor="signature-based">Signature-based detection</Label>
                    </div>
                    <Badge className="bg-primary/20 text-primary border-primary/10">
                      Active
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Uses known patterns to identify threats
                  </p>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Switch defaultChecked id="anomaly-detection" />
                      <Label htmlFor="anomaly-detection">Anomaly detection</Label>
                    </div>
                    <Badge className="bg-primary/20 text-primary border-primary/10">
                      Active
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Identifies deviations from normal behavior
                  </p>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Switch defaultChecked id="ai-detection" />
                      <Label htmlFor="ai-detection">AI-powered detection</Label>
                    </div>
                    <Badge className="bg-accent/20 text-accent border-accent/10">
                      Premium
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Uses machine learning algorithms to identify complex threats
                  </p>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button onClick={saveSettings}>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Notifications Settings */}
        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle>Alert & Notification Settings</CardTitle>
              <CardDescription>
                Configure how you receive alerts and notifications
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Notification Channels</h3>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={dashboardNotifications}
                        onCheckedChange={setDashboardNotifications}
                        id="dashboard-notifications"
                      />
                      <Label htmlFor="dashboard-notifications">Dashboard notifications</Label>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Show alerts in the dashboard interface
                  </p>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={emailNotifications}
                        onCheckedChange={setEmailNotifications}
                        id="email-notifications"
                      />
                      <Label htmlFor="email-notifications">Email notifications</Label>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Send alert emails to specified addresses
                  </p>
                  {emailNotifications && (
                    <div className="mt-2 space-y-2">
                      <Label htmlFor="email-addresses">Email addresses</Label>
                      <Input id="email-addresses" placeholder="admin@example.com, security@example.com" />
                    </div>
                  )}
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={smsNotifications}
                        onCheckedChange={setSmsNotifications}
                        id="sms-notifications"
                      />
                      <Label htmlFor="sms-notifications">SMS notifications</Label>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Send text message alerts for critical events
                  </p>
                  {smsNotifications && (
                    <div className="mt-2 space-y-2">
                      <Label htmlFor="phone-numbers">Phone numbers</Label>
                      <Input id="phone-numbers" placeholder="+1234567890, +0987654321" />
                    </div>
                  )}
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Switch id="webhook-notifications" />
                      <Label htmlFor="webhook-notifications">Webhook notifications</Label>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Send notifications to external systems via webhooks
                  </p>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Alert Severity Settings</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <AlertCircle className="h-4 w-4 text-accent" />
                      <Label>Critical</Label>
                    </div>
                    <Select defaultValue="all">
                      <SelectTrigger>
                        <SelectValue placeholder="Select channels" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All channels</SelectItem>
                        <SelectItem value="email-sms">Email & SMS</SelectItem>
                        <SelectItem value="dashboard-email">Dashboard & Email</SelectItem>
                        <SelectItem value="dashboard">Dashboard only</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <AlertCircle className="h-4 w-4 text-yellow-500" />
                      <Label>Warning</Label>
                    </div>
                    <Select defaultValue="dashboard-email">
                      <SelectTrigger>
                        <SelectValue placeholder="Select channels" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All channels</SelectItem>
                        <SelectItem value="email-sms">Email & SMS</SelectItem>
                        <SelectItem value="dashboard-email">Dashboard & Email</SelectItem>
                        <SelectItem value="dashboard">Dashboard only</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <AlertCircle className="h-4 w-4 text-blue-500" />
                      <Label>Info</Label>
                    </div>
                    <Select defaultValue="dashboard">
                      <SelectTrigger>
                        <SelectValue placeholder="Select channels" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All channels</SelectItem>
                        <SelectItem value="email-sms">Email & SMS</SelectItem>
                        <SelectItem value="dashboard-email">Dashboard & Email</SelectItem>
                        <SelectItem value="dashboard">Dashboard only</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button onClick={saveSettings}>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Network Settings */}
        <TabsContent value="network">
          <Card>
            <CardHeader>
              <CardTitle>Network Settings</CardTitle>
              <CardDescription>
                Configure network monitoring and detection settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Network Discovery</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="network-range">Network Range</Label>
                    <Input id="network-range" defaultValue="192.168.1.0/24" />
                  </div>
                  <div className="space-y-2">
                    <Label>Discovery Frequency</Label>
                    <Select defaultValue="hourly">
                      <SelectTrigger>
                        <SelectValue placeholder="Select frequency" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="continuous">Continuous</SelectItem>
                        <SelectItem value="hourly">Hourly</SelectItem>
                        <SelectItem value="daily">Daily</SelectItem>
                        <SelectItem value="weekly">Weekly</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="auto-classify" />
                  <Label htmlFor="auto-classify">Automatically classify new devices</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="alert-new-device" />
                  <Label htmlFor="alert-new-device">Alert on new device discovery</Label>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium">Protected Networks</h3>
                  <Button variant="outline" size="sm">Add Network</Button>
                </div>
                
                <div className="space-y-2">
                  {[
                    { name: "Main Network", range: "192.168.1.0/24", status: "active" },
                    { name: "Guest Network", range: "192.168.2.0/24", status: "active" },
                    { name: "IoT Network", range: "192.168.3.0/24", status: "active" }
                  ].map((network, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-secondary/30 rounded">
                      <div>
                        <div className="font-medium">{network.name}</div>
                        <div className="text-xs text-muted-foreground">{network.range}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="bg-green-500/20 text-green-400">
                          Monitoring {network.status === "active" ? "Active" : "Paused"}
                        </Badge>
                        <Button variant="ghost" size="sm">Edit</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Advanced Settings</h3>
                
                <div className="space-y-2">
                  <Label>Traffic Monitoring</Label>
                  <Select defaultValue="headers">
                    <SelectTrigger>
                      <SelectValue placeholder="Select mode" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="headers">Packet headers only</SelectItem>
                      <SelectItem value="metadata">Headers and metadata</SelectItem>
                      <SelectItem value="full">Full packet inspection</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Determines how much of each packet is inspected. Higher levels use more system resources.
                  </p>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="capture-pcap" />
                  <Label htmlFor="capture-pcap">Capture PCAP for suspicious traffic</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="protocol-anomalies" />
                  <Label htmlFor="protocol-anomalies">Detect protocol anomalies</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="dns-monitoring" />
                  <Label htmlFor="dns-monitoring">Enable DNS monitoring</Label>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button onClick={saveSettings}>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* User Management Settings */}
        <TabsContent value="users">
          <Card>
            <CardHeader>
              <CardTitle>User Management</CardTitle>
              <CardDescription>
                Manage system users and access controls
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium">System Users</h3>
                <Button variant="outline" size="sm">Add User</Button>
              </div>
              
              <div className="space-y-2">
                {[
                  { name: "Administrator", email: "admin@example.com", role: "admin", lastLogin: "Today, 9:15 AM" },
                  { name: "Security Analyst", email: "security@example.com", role: "analyst", lastLogin: "Yesterday, 5:20 PM" },
                  { name: "Network Monitor", email: "monitor@example.com", role: "viewer", lastLogin: "3 days ago" }
                ].map((user, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-secondary/30 rounded">
                    <div className="flex items-center gap-3">
                      <div className="h-9 w-9 rounded-full bg-primary/20 flex items-center justify-center">
                        <User size={16} className="text-primary" />
                      </div>
                      <div>
                        <div className="font-medium">{user.name}</div>
                        <div className="text-xs text-muted-foreground">{user.email}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div>
                        <Badge variant="outline" className={
                          user.role === 'admin' 
                            ? 'bg-primary/20 text-primary' 
                            : user.role === 'analyst'
                              ? 'bg-blue-500/20 text-blue-400'
                              : 'bg-secondary/50 text-secondary-foreground'
                        }>
                          {user.role === 'admin' ? 'Administrator' : user.role === 'analyst' ? 'Security Analyst' : 'Viewer'}
                        </Badge>
                        <div className="text-xs text-muted-foreground mt-1">Last login: {user.lastLogin}</div>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="ghost" size="sm">Edit</Button>
                        <Button variant="ghost" size="sm" className="text-destructive">Delete</Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <Separator />
              
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Access Controls</h3>
                
                <div className="space-y-2">
                  <Label>Authentication Method</Label>
                  <Select defaultValue="local">
                    <SelectTrigger>
                      <SelectValue placeholder="Select method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="local">Local authentication</SelectItem>
                      <SelectItem value="ldap">LDAP / Active Directory</SelectItem>
                      <SelectItem value="saml">SAML</SelectItem>
                      <SelectItem value="oauth">OAuth</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Password Policy</Label>
                  <Select defaultValue="strong">
                    <SelectTrigger>
                      <SelectValue placeholder="Select policy" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="basic">Basic (8+ characters)</SelectItem>
                      <SelectItem value="standard">Standard (8+ chars, mixed case, numbers)</SelectItem>
                      <SelectItem value="strong">Strong (12+ chars, mixed case, numbers, symbols)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Session Timeout (minutes)</Label>
                  <Input type="number" defaultValue="30" />
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="enforce-2fa" />
                  <Label htmlFor="enforce-2fa">Enforce two-factor authentication</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="account-lockout" />
                  <Label htmlFor="account-lockout">Enable account lockout after failed attempts</Label>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Audit Settings</h3>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="log-user-actions" />
                  <Label htmlFor="log-user-actions">Log all user actions</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="log-auth-attempts" />
                  <Label htmlFor="log-auth-attempts">Log authentication attempts</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch defaultChecked id="log-settings-changes" />
                  <Label htmlFor="log-settings-changes">Log settings changes</Label>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button onClick={saveSettings}>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Settings;
