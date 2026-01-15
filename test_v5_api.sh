#!/bin/bash
# LegoMCP v5.0 API Test Script
# Tests all major API endpoints to verify they return JSON

BASE_URL="http://localhost:5000"
PASS=0
FAIL=0

test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint" 2>/dev/null)
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')

    if [[ "$http_code" == "200" ]] && echo "$body" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
        echo "✅ $name ($endpoint)"
        ((PASS++))
    else
        echo "❌ $name ($endpoint) - HTTP $http_code"
        ((FAIL++))
    fi
}

echo "=============================================="
echo "LegoMCP v5.0 API Test Suite"
echo "=============================================="
echo ""

echo "--- Phase 1: Health & Core ---"
test_endpoint "API Health" "/api/health"
test_endpoint "v5 Health" "/api/v5/health"
test_endpoint "Catalog" "/api/catalog"

echo ""
echo "--- Phase 2: MES (Manufacturing) ---"
test_endpoint "Work Centers" "/api/mes/work-centers"
test_endpoint "Shop Floor Dashboard" "/api/mes/shop-floor/dashboard"
test_endpoint "Shop Floor Queue" "/api/mes/shop-floor/queue"
test_endpoint "Active Operations" "/api/mes/shop-floor/active-operations"
test_endpoint "Andon Display" "/api/mes/shop-floor/andon"

echo ""
echo "--- Phase 3: Quality ---"
test_endpoint "SPC Dashboard" "/api/quality/spc/dashboard"
test_endpoint "LEGO Specs" "/api/quality/lego/specs"

echo ""
echo "--- Phase 4: ERP ---"
test_endpoint "Suppliers" "/api/erp/procurement/suppliers"
test_endpoint "Purchase Orders" "/api/erp/procurement/purchase-orders"

echo ""
echo "--- Phase 5: MRP ---"
test_endpoint "Capacity Overview" "/api/mrp/capacity/overview"
test_endpoint "Capacity Bottlenecks" "/api/mrp/capacity/bottlenecks"

echo ""
echo "=============================================="
echo "RESULTS: $PASS passed, $FAIL failed"
echo "=============================================="

if [ $FAIL -gt 0 ]; then
    exit 1
fi
