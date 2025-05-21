
import { ReactNode } from "react";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { Bell, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        <AppSidebar />
        <div className="flex-1 flex flex-col">
          <header className="h-16 border-b bg-background/95 backdrop-blur flex items-center justify-between px-6">
            <h1 className="text-lg font-semibold">Network Defense System</h1>
            <div className="flex items-center gap-4">
              <div className="relative">
                <Button variant="ghost" size="icon">
                  <Bell size={18} />
                </Button>
                <Badge className="absolute -top-1 -right-1 px-1.5 py-0.5 bg-accent text-accent-foreground">
                  3
                </Badge>
              </div>
              <Button variant="ghost" size="icon">
                <User size={18} />
              </Button>
            </div>
          </header>
          <main className="flex-1 overflow-auto p-6">
            {children}
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}

export default MainLayout;
