import argparse
import os
import json
from pathlib import Path

# Mock data for demo purposes (replace or load dynamically if needed)
ARCHS = [
    'aarch64', 'armbe', 'armle', 'cbea', 'cbea64', 'cmd', 'dalvik', 'firefox', 'java', 'mips', 'mips64',
    'mips64le', 'mipsbe', 'mipsle', 'nodejs', 'php', 'ppc', 'ppc64', 'ppc64le', 'ppce500v2', 'python',
    'r', 'ruby', 'sparc', 'sparc64', 'tty', 'x64', 'x86', 'x86_64', 'zarch'
]

FORMATS = [
    'asp', 'aspx', 'aspx-exe', 'axis2', 'dll', 'ducky-script-psh', 'elf', 'elf-so', 'exe', 'exe-only',
    'exe-service', 'exe-small', 'hta-psh', 'jar', 'jsp', 'loop-vbs', 'macho', 'msi', 'msi-nouac',
    'osx-app', 'psh', 'psh-cmd', 'psh-net', 'psh-reflection', 'vba', 'vba-exe', 'vba-psh', 'vbs', 'war',
    'bash', 'c', 'csharp', 'dw', 'dword', 'hex', 'java', 'js_be', 'js_le', 'num', 'perl', 'pl',
    'powershell', 'ps1', 'py', 'python', 'raw', 'rb', 'ruby', 'sh', 'vbapplication', 'vbscript'
]

ENCODERS = [
    'cmd/brace', 'cmd/echo', 'cmd/generic_sh', 'cmd/ifs', 'cmd/perl', 'cmd/powershell_base64',
    'generic/eicar', 'generic/none', 'mipsbe/byte_xori', 'mipsbe/longxor', 'php/base64',
    'ppc/longxor', 'ruby/base64', 'x64/xor', 'x86/shikata_ga_nai'
]

PLATFORMS = [
    'aix', 'android', 'apple_ios', 'bsd', 'cisco', 'firefox', 'freebsd', 'hardware',
    'hpux', 'irix', 'java', 'javascript', 'linux', 'mainframe', 'multi', 'netbsd',
    'netware', 'nodejs', 'openbsd', 'osx', 'php', 'python', 'r', 'ruby', 'solaris', 'unix', 'windows'
]

def load_payloads(cache_path=None):
    default_path = Path.home() / ".msf4/store/modules_metadata.json"
    if cache_path is None:
        cache_path = default_path
    if not cache_path.exists():
        return ["windows/meterpreter/reverse_tcp:Reverse shell for Windows (mock)"]

    with open(cache_path, "r") as f:
        data = json.load(f)

    payloads = []
    for mod in data.get("modules", []):
        if mod.get("type") == "payload":
            name = mod.get("ref_name", "unknown")
            desc = mod.get("description", "")
            payloads.append(f"{name}:{desc}")
    return payloads

def main():
    parser = argparse.ArgumentParser(description="msfvenom clone tool in Python")

    parser.add_argument("-p", "--payload", help="Payload to use", choices=load_payloads())
    parser.add_argument("-f", "--format", help="Output format", choices=FORMATS)
    parser.add_argument("-a", "--arch", help="Architecture", choices=ARCHS)
    parser.add_argument("-e", "--encoder", help="Encoder", choices=ENCODERS)
    parser.add_argument("--platform", help="Platform", choices=PLATFORMS)
    parser.add_argument("-o", "--out", help="Output file")
    parser.add_argument("-s", "--space", help="Max size of payload", type=int)
    parser.add_argument("-i", "--iterations", help="Encoding iterations", type=int)
    parser.add_argument("-n", "--nopsled", help="Nopsled length", type=int)
    parser.add_argument("-x", "--template", help="Custom template executable")
    parser.add_argument("--smallest", action="store_true", help="Use smallest possible payload")
    parser.add_argument("--sec-name", help="Windows section name")
    parser.add_argument("--encrypt", help="Type of encryption")
    parser.add_argument("--encrypt-key", help="Encryption key")
    parser.add_argument("--encrypt-iv", help="Encryption IV")
    parser.add_argument("-k", "--keep", action="store_true", help="Preserve template behavior")
    parser.add_argument("-v", "--var-name", help="Custom variable name")
    parser.add_argument("-t", "--timeout", help="Timeout reading STDIN", type=int, default=30)
    parser.add_argument("--list", choices=["payloads", "encoders", "platforms", "formats", "archs", "all"],
                        help="List available options")

    args = parser.parse_args()

    # Handle --list
    if args.list:
        if args.list == "payloads":
            print("\n".join(load_payloads()))
        elif args.list == "encoders":
            print("\n".join(ENCODERS))
        elif args.list == "platforms":
            print("\n".join(PLATFORMS))
        elif args.list == "formats":
            print("\n".join(FORMATS))
        elif args.list == "archs":
            print("\n".join(ARCHS))
        elif args.list == "all":
            print("Payloads:", load_payloads())
            print("Encoders:", ENCODERS)
            print("Platforms:", PLATFORMS)
            print("Formats:", FORMATS)
            print("Architectures:", ARCHS)
        return

    # Simulate output
    print(f"\n[*] Generating payload with:")
    for arg, val in vars(args).items():
        if val:
            print(f"  - {arg}: {val}")

    print("\n[+] Done! (This is a simulation. Implement actual shellcode logic here.)")

if __name__ == "__main__":
    main()
