# Security

cuda-translator parses untrusted binary/text formats and can generate and run
OpenCL kernels. Treat CUDA artifacts from untrusted sources as potentially
hostile input.

The REST server is intentionally small and has **no authentication or TLS**.
It binds to `127.0.0.1` by default. Do not expose it directly to an untrusted
network; place an authenticated, rate-limited boundary in front of it if remote
access is required.

The MCP server uses stdio and inherits the trust boundary of the process that
launches it. The translation pipeline is fail-closed for unsupported semantics,
but that is a correctness property, not a sandbox. GPU compilation/execution
can consume substantial CPU, memory, driver, or GPU resources.

For security-sensitive use, run the translator with least privilege and isolate
it from secrets and unrelated writable data. Keep GPU/OpenCL drivers current.

If you find a vulnerability, use GitHub's private security-reporting/advisory
flow when it is available for this repository. If private reporting is not
available, contact the maintainer privately through the GitHub profile rather
than publishing exploit details in a public issue.
