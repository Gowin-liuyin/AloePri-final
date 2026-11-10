"""生成本地TLS测试证书与服务访问令牌；不修改系统证书信任设置。"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import ipaddress
import secrets


def create_local_credentials(directory,*,days=30,dns_names=("localhost",),ip_addresses=("127.0.0.1",)):
    from cryptography import x509
    from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    root=Path(directory)
    if type(days) is not int or not 1<=days<=365:
        raise ValueError("Certificate validity must be 1-365 days")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Credential directory must be empty")
    names=[x509.DNSName(n) for n in dns_names]+[x509.IPAddress(ipaddress.ip_address(n)) for n in ip_addresses]
    if not names:raise ValueError("At least one certificate hostname/IP is required")
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    subject=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,"AloePri local service")])
    now=datetime.now(timezone.utc)
    certificate=(x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
       .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=days))
       .add_extension(x509.SubjectAlternativeName(names),critical=False)
       .add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True)
       .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),critical=False).sign(key,hashes.SHA256()))
    root.mkdir(parents=True,exist_ok=True)
    contents={"server.key":key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()),
              "server.crt":certificate.public_bytes(serialization.Encoding.PEM),"auth_token.txt":(secrets.token_urlsafe(32)+"\n").encode()}
    for name,data in contents.items():
        path=root/name
        with path.open("xb") as handle:handle.write(data)
        if name!="server.crt":path.chmod(0o600)
    return {name:str(root/name) for name in contents}
