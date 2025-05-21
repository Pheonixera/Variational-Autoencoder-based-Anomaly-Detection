
import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

// Mock network data
const networkData = {
  nodes: [
    { id: "firewall", group: 1, name: "Firewall", status: "healthy" },
    { id: "router", group: 1, name: "Router", status: "healthy" },
    { id: "server1", group: 2, name: "Web Server", status: "healthy" },
    { id: "server2", group: 2, name: "Database", status: "warning" },
    { id: "server3", group: 2, name: "Authentication", status: "healthy" },
    { id: "client1", group: 3, name: "Client 1", status: "healthy" },
    { id: "client2", group: 3, name: "Client 2", status: "healthy" },
    { id: "client3", group: 3, name: "Client 3", status: "critical" },
    { id: "client4", group: 3, name: "Client 4", status: "healthy" },
    { id: "client5", group: 3, name: "Client 5", status: "healthy" },
    { id: "device1", group: 4, name: "IoT Device 1", status: "healthy" },
    { id: "device2", group: 4, name: "IoT Device 2", status: "warning" }
  ],
  links: [
    { source: "firewall", target: "router", value: 10 },
    { source: "router", target: "server1", value: 8 },
    { source: "router", target: "server2", value: 8 },
    { source: "router", target: "server3", value: 8 },
    { source: "server1", target: "client1", value: 5, status: "active" },
    { source: "server1", target: "client2", value: 5 },
    { source: "server2", target: "client3", value: 5, status: "active" },
    { source: "server2", target: "client4", value: 5 },
    { source: "server3", target: "client5", value: 5 },
    { source: "router", target: "device1", value: 3 },
    { source: "router", target: "device2", value: 3, status: "active" }
  ]
};

const NetworkGraph: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    
    // Clear any existing SVG contents
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    // Create a force simulation
    const simulation = d3.forceSimulation(networkData.nodes)
      .force("link", d3.forceLink(networkData.links).id((d: any) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30));

    // Draw links
    const link = svg.append("g")
      .selectAll("line")
      .data(networkData.links)
      .enter()
      .append("line")
      .attr("class", (d: any) => `network-link ${d.status || ""}`)
      .attr("stroke-width", (d: any) => Math.sqrt(d.value));

    // Create a group for each node
    const nodeGroup = svg.append("g")
      .selectAll("g")
      .data(networkData.nodes)
      .enter()
      .append("g")
      .call(d3.drag<SVGGElement, any>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add circles to each node group
    nodeGroup.append("circle")
      .attr("r", (d: any) => getNodeRadius(d))
      .attr("class", "network-node")
      .attr("fill", (d: any) => getNodeColor(d.status));

    // Add text labels to each node
    nodeGroup.append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#ffffff")
      .style("font-size", "10px")
      .style("pointer-events", "none")
      .text((d: any) => getNodeLabel(d));

    // Add tooltips
    nodeGroup.append("title")
      .text((d: any) => `${d.name}\nStatus: ${d.status}`);

    // Update positions on each tick of the simulation
    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      nodeGroup
        .attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    // Helper functions
    function getNodeRadius(node: any) {
      switch (node.group) {
        case 1: return 25; // Firewall, router
        case 2: return 20; // Servers
        default: return 15; // Clients, devices
      }
    }

    function getNodeColor(status: string) {
      switch (status) {
        case 'critical': return '#F97316'; // Orange for critical
        case 'warning': return '#FBBF24'; // Yellow for warning
        default: return '#0EA5E9'; // Blue for healthy
      }
    }

    function getNodeLabel(node: any) {
      // Only show short labels
      return node.id.substring(0, 3);
    }

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, []);

  return (
    <div className="w-full h-[500px] bg-card rounded-lg p-4 overflow-hidden">
      <svg ref={svgRef} className="w-full h-full"></svg>
    </div>
  );
};

export default NetworkGraph;
