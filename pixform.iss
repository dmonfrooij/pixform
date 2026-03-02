; PIXFORM - Inno Setup Script

#define MyAppName "PIXFORM"
#define MyAppVersion "1.0"
#define MyAppPublisher "PIXFORM"
#define MyAppURL "https://github.com/YOUR_USERNAME/pixform"
#define MyAppExeName "PIXFORM.bat"

[Setup]
AppId={{A7B3C2D1-E4F5-6789-ABCD-EF0123456789}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=dist
OutputBaseFilename=PIXFORM_Setup_v1.0
Compression=lzma
SolidCompression=yes
WizardStyle=modern
MinVersion=10.0
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
Name: "startmenu";   Description: "Create a Start Menu shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "backend\*";  DestDir: "{app}\backend";  Flags: ignoreversion recursesubdirs createallsubdirs
Source: "frontend\*"; DestDir: "{app}\frontend"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "PIXFORM.bat"; DestDir: "{app}";         Flags: ignoreversion
Source: "install.ps1"; DestDir: "{app}";         Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";                 Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Install PIXFORM (first time)"; Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File ""{app}\install.ps1"""; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}";       Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}";         Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartmenu}\{#MyAppName}";         Filename: "{app}\{#MyAppExeName}"; Tasks: startmenu

[Run]
Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\install.ps1"""; \
  WorkingDir: "{app}"; \
  Description: "Install Python dependencies (required, ~10-20 min)"; \
  Flags: postinstall nowait; \
  StatusMsg: "Launching PIXFORM installer..."

[Code]
procedure InitializeWizard;
begin
  WizardForm.WelcomeLabel2.Caption :=
    'This will install PIXFORM on your computer.' + #13#10 + #13#10 +
    'PIXFORM converts images to 3D models using AI.' + #13#10 + #13#10 +
    'After installation, a second window will open to' + #13#10 +
    'download Python dependencies (~5-8 GB).' + #13#10 +
    'This takes 10-20 minutes on first run.' + #13#10 + #13#10 +
    'Requirements:' + #13#10 +
    '  - Python 3.10 (checked during install)' + #13#10 +
    '  - NVIDIA GPU with 6+ GB VRAM' + #13#10 +
    '  - Internet connection for first-time setup' + #13#10 + #13#10 +
    'Click Next to continue.';
end;
