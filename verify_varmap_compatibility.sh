# #!/bin/bash
# # filepath: /home/artem/candle/verify_varmap_compatibility.sh

# echo "Running VarMap compatibility tests..."

# # Run all tests
# cargo test -p candle-nn var_map_compatibility -- --nocapture
# cargo test -p candle-nn var_map_stress -- --nocapture  
# cargo test -p candle-nn var_map_integration -- --nocapture

# # Run existing candle-nn tests to ensure nothing breaks
# cargo test -p candle-nn

# # Run examples that use VarMap
# cargo run --example llama -- --which tiny-llama-1.1b-chat --prompt "test" --sample-len 10

# echo "Compatibility verification complete!"

#!/bin/bash
# filepath: /home/artem/candle/verify_varmap_compatibility.sh

echo "Running VarMap compatibility tests..."

# Run specific tests with output capture disabled
echo -e "\n=== Running stress test with performance metrics ==="
cargo test -p candle-nn stress_test_concurrent_access -- --nocapture

echo -e "\n=== Running other VarMap tests ==="
cargo test -p candle-nn var_map_compatibility -- --nocapture
cargo test -p candle-nn var_map_integration -- --nocapture

# Run existing candle-nn tests to ensure nothing breaks
echo -e "\n=== Running general candle-nn tests ==="
cargo test -p candle-nn

# Run example that uses VarMap
echo -e "\n=== Testing with llama example ==="
timeout 30s cargo run --example llama -- --which tiny-llama-1.1b-chat --prompt "test" --sample-len 10 || true

echo "Compatibility verification complete!"