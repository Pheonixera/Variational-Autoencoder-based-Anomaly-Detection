
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarTrigger,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { Link, useLocation } from "react-router-dom";
import { 
  ActivitySquare, 
  AlertTriangle, 
  Gauge, 
  History, 
  Settings, 
  ShieldCheck 
} from "lucide-react";

export function AppSidebar() {
  const location = useLocation();

  const menuItems = [
    {
      title: "Dashboard",
      path: "/",
      icon: ActivitySquare,
    },
    {
      title: "Anomalies",
      path: "/anomalies",
      icon: AlertTriangle, 
    },
    {
      title: "Risk Meter",
      path: "/risk-meter",
      icon: Gauge,
    },
    {
      title: "History",
      path: "/history",
      icon: History,
    },
    {
      title: "Settings",
      path: "/settings",
      icon: Settings,
    },
  ];

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <Sidebar>
      <SidebarHeader className="p-4">
        <div className="flex items-center gap-2 px-2">
          <ShieldCheck className="h-8 w-8 text-primary" />
          <div className="font-bold text-2xl">SENTINEL</div>
          <div className="ml-auto">
            <SidebarTrigger />
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Main Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.path}>
                  <SidebarMenuButton asChild active={isActive(item.path)}>
                    <Link to={item.path}>
                      <item.icon size={20} />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="p-4 text-xs text-muted-foreground">
        SENTINEL IDS v1.0.0
      </SidebarFooter>
    </Sidebar>
  );
}

export default AppSidebar;
