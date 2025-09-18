#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
BUILD=false
if [[ "$1" == "--build" ]] || [[ "$1" == "-b" ]]; then
    BUILD=true
fi

echo -e "${BLUE}🚀 Relay Full Stack Setup${NC}"
echo -e "${BLUE}=========================${NC}"
echo
if [ "$BUILD" = true ]; then
    echo -e "${GREEN}🔨 Build mode enabled - will rebuild images${NC}"
    echo
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}📝 Creating .env file from template...${NC}"
    cp .env.template .env
fi

if [ "$BUILD" = true ]; then
    echo -e "${YELLOW}🔨 Building and Starting Relay Full Stack...${NC}"
    echo -e "   This will build and start:"
else
    echo -e "${YELLOW}🔧 Starting Relay Full Stack...${NC}"
    echo -e "   This will start:"
fi
echo -e "   - 🗄️  PostgreSQL Database"
echo -e "   - 🗃️  Dragonfly Cache"
echo -e "   - ⚙️  Relay API Server"
echo -e "   - 🌐 Relay Frontend"
echo

# Start services (with optional build)
if [ "$BUILD" = true ]; then
    echo -e "${BLUE}🔨 Building images...${NC}"
    docker-compose -f docker-compose.full-stack.yml up -d --build
else
    docker-compose -f docker-compose.full-stack.yml up -d
fi

echo
echo -e "${GREEN}✅ Relay Full Stack is starting up!${NC}"
echo
echo -e "${BLUE}📱 Access Points:${NC}"
echo -e "   🌐 Chat Interface:    ${GREEN}http://localhost:3000${NC}"
echo -e "   📖 API Documentation: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "   🔍 API Explorer:      ${GREEN}http://localhost:8000/redoc${NC}"
echo
echo -e "${YELLOW}⏳ Note: It may take 1-2 minutes for all services to be ready${NC}"
echo
echo -e "${BLUE}🔑 Configure LLM Providers:${NC}"
echo -e "   Configure your API keys through the provider section in the database"
echo -e "   or use relay-connect for provider management"
echo
echo -e "${BLUE}🛫 To stop everything:${NC}"
echo -e "   docker-compose -f docker-compose.full-stack.yml down"
echo
echo -e "${BLUE}📚 Usage:${NC}"
echo -e "   $0           # Start with existing/pulled images"
echo -e "   $0 --build   # Build images from source (use -b as shorthand)"
echo
