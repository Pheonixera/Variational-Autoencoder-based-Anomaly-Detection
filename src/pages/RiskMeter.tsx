
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { AlertTriangle, ShieldAlert, ShieldCheck, Gauge, Info, Server, Laptop, Wifi } from "lucide-react";

const RiskMeter = () => {
  // Mock risk data
  const currentRisk = 45;
  const riskLevel = "medium";
  
  const getRiskColor = () => {
    if (currentRisk < 25) return "bg-green-500";
    if (currentRisk < 50) return "bg-yellow-500";
    if (currentRisk < 75) return "bg-orange-500";
    return "bg-red-500";
  };
  
  const getRiskIcon = () => {
    if (currentRisk < 25) return <ShieldCheck className="h-8 w-8 text-green-500" />;
    if (currentRisk < 50) return <AlertTriangle className="h-8 w-8 text-yellow-500" />;
    if (currentRisk < 75) return <ShieldAlert className="h-8 w-8 text-orange-500" />;
    return <ShieldAlert className="h-8 w-8 text-red-500" />;
  };
  
  const categoryRisks = [
    { name: "Network Infrastructure", level: 35, icon: Server },
    { name: "Client Devices", level: 60, icon: Laptop },
    { name: "Authentication Systems", level: 25, icon: ShieldCheck },
    { name: "Wireless Networks", level: 55, icon: Wifi }
  ];
  
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Risk Assessment</h2>
        <div className="flex items-center gap-2">
          <Info size={14} className="text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Last updated: 15 minutes ago</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Overall Security Risk</CardTitle>
            <CardDescription>Current system-wide security risk assessment</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center">
            <div className="w-64 h-64 relative mb-6">
              <div className="gauge-container">
                <svg className="w-full h-full" viewBox="0 0 120 120">
                  {/* Background ring */}
                  <circle
                    cx="60"
                    cy="60"
                    r="54"
                    fill="none"
                    stroke="#374151"
                    strokeWidth="12"
                  />
                  {/* Colored progress arc */}
                  <circle
                    cx="60"
                    cy="60"
                    r="54"
                    fill="none"
                    stroke={getRiskColor().replace('bg-', 'stroke-')}
                    strokeWidth="12"
                    strokeLinecap="round"
                    strokeDasharray={`${currentRisk * 3.39} 1000`}
                    style={{
                      transformOrigin: 'center',
                      transform: 'rotate(-90deg)',
                    }}
                  />
                  {/* Center circle */}
                  <circle
                    cx="60"
                    cy="60"
                    r="42"
                    fill="#1a1f2c"
                    stroke="#1a1f2c"
                    strokeWidth="2"
                  />
                  
                  {/* Risk indicator */}
                  <text x="60" y="55" 
                    textAnchor="middle" 
                    fontSize="24" 
                    fontWeight="bold" 
                    fill="white"
                  >
                    {currentRisk}%
                  </text>
                  <text x="60" y="75" 
                    textAnchor="middle" 
                    fontSize="14"  
                    fill="white"
                    style={{ textTransform: 'capitalize' }}
                  >
                    {riskLevel}
                  </text>
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  {getRiskIcon()}
                </div>
              </div>
            </div>
            
            <div className="w-full max-w-md grid grid-cols-2 gap-4 mt-4">
              <div className="flex justify-between p-2 bg-secondary/30 rounded">
                <span className="text-sm">Active Threats:</span>
                <span className="font-semibold">3</span>
              </div>
              <div className="flex justify-between p-2 bg-secondary/30 rounded">
                <span className="text-sm">Vulnerabilities:</span>
                <span className="font-semibold">7</span>
              </div>
              <div className="flex justify-between p-2 bg-secondary/30 rounded">
                <span className="text-sm">Past 24h Change:</span>
                <span className="font-semibold text-accent">+5%</span>
              </div>
              <div className="flex justify-between p-2 bg-secondary/30 rounded">
                <span className="text-sm">Weekly Trend:</span>
                <span className="font-semibold text-green-500">-2%</span>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Risk by Category</CardTitle>
            <CardDescription>Breakdown of risk levels across system categories</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {categoryRisks.map((category) => (
              <div key={category.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <category.icon className="h-4 w-4 text-muted-foreground" />
                    <span>{category.name}</span>
                  </div>
                  <span 
                    className={`text-sm font-medium ${
                      category.level < 25 
                        ? 'text-green-500' 
                        : category.level < 50 
                          ? 'text-yellow-500' 
                          : category.level < 75 
                            ? 'text-orange-500' 
                            : 'text-red-500'
                    }`}
                  >
                    {category.level}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-secondary overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${
                      category.level < 25 
                        ? 'bg-green-500' 
                        : category.level < 50 
                          ? 'bg-yellow-500' 
                          : category.level < 75 
                            ? 'bg-orange-500' 
                            : 'bg-red-500'
                    }`} 
                    style={{ width: `${category.level}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Risk Factors</CardTitle>
            <CardDescription>Key elements affecting current risk level</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                {
                  name: "Multiple Authentication Failures",
                  impact: "high",
                  description: "Repeated login failures on Database Server"
                },
                {
                  name: "Unpatched Vulnerabilities",
                  impact: "medium",
                  description: "3 systems with pending security updates"
                },
                {
                  name: "Suspicious Network Traffic",
                  impact: "high",
                  description: "Unusual outbound connection patterns detected"
                },
                {
                  name: "Weak Password Policies",
                  impact: "low",
                  description: "Some user accounts using common passwords"
                }
              ].map((factor, index) => (
                <div key={index} className="p-3 bg-secondary/30 rounded border border-secondary flex justify-between">
                  <div>
                    <div className="font-medium">{factor.name}</div>
                    <div className="text-sm text-muted-foreground">{factor.description}</div>
                  </div>
                  <div>
                    <span className={`text-xs px-2 py-1 rounded ${
                      factor.impact === 'high' 
                        ? 'bg-red-500/20 text-red-400' 
                        : factor.impact === 'medium'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-green-500/20 text-green-400'
                    }`}>
                      {factor.impact} impact
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Recommended Actions</CardTitle>
          <CardDescription>Steps to mitigate current risks</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              {
                title: "Update Database Server",
                priority: "high",
                description: "Apply latest security patches to database server"
              },
              {
                title: "Investigate Client 3",
                priority: "high",
                description: "Examine unusual traffic patterns on Client 3"
              },
              {
                title: "Enforce Password Policy",
                priority: "medium",
                description: "Require password changes for accounts with weak credentials"
              },
              {
                title: "Update IDS Signatures",
                priority: "medium",
                description: "Download and apply latest threat signatures"
              }
            ].map((action, index) => (
              <div 
                key={index} 
                className={`p-4 rounded border ${
                  action.priority === 'high' 
                    ? 'border-red-800 bg-red-950/20' 
                    : action.priority === 'medium'
                      ? 'border-yellow-800 bg-yellow-950/20'
                      : 'border-green-800 bg-green-950/20'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-semibold">{action.title}</h4>
                  <span className={`text-xs px-2 py-1 rounded capitalize ${
                    action.priority === 'high' 
                      ? 'bg-red-500/20 text-red-400' 
                      : action.priority === 'medium'
                        ? 'bg-yellow-500/20 text-yellow-400'
                        : 'bg-green-500/20 text-green-400'
                  }`}>
                    {action.priority} priority
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">{action.description}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default RiskMeter;
