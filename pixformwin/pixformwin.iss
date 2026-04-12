; PIXFORMWIN - Inno Setup Script
; Builds a one-click Windows installer (.exe)

#define MyAppName "PIXFORMWIN"
#define MyAppVersion "1.0"
#define MyAppPublisher "PIXFORMWIN"
#define MyAppURL "https://github.com/dmonfrooij/pixformwin"
#define MyAppExeName "PIXFORMWIN.bat"

[Setup]
AppId={{B8C4D3E2-F5A6-7890-BCDE-F01234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=dist
OutputBaseFilename=PIXFORMWIN_Setup_v1.0
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
Source: "app.py";     DestDir: "{app}";           Flags: ignoreversion
Source: "backend\*";  DestDir: "{app}\backend";   Flags: ignoreversion recursesubdirs createallsubdirs
Source: "frontend\*"; DestDir: "{app}\frontend";  Flags: ignoreversion recursesubdirs createallsubdirs
Source: "PIXFORMWIN.bat"; DestDir: "{app}";        Flags: ignoreversion
Source: "install.ps1";    DestDir: "{app}";        Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";                    Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Install PIXFORMWIN (first time)"; Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File ""{app}\install.ps1"""; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}";          Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}";            Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartmenu}\{#MyAppName}";            Filename: "{app}\{#MyAppExeName}"; Tasks: startmenu

[Run]
Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\install.ps1"""; \
  WorkingDir: "{app}"; \
  Description: "Install Python dependencies (required, ~10-20 min)"; \
  Flags: postinstall nowait; \
  StatusMsg: "Launching PIXFORMWIN installer..."

[Code]
procedure InitializeWizard;
begin
  WizardForm.WelcomeLabel2.Caption :=
    'This will install PIXFORMWIN on your computer.' + #13#10 + #13#10 +
    'PIXFORMWIN converts images to 3D models using AI.' + #13#10 +
    'It runs as a native desktop window — no browser needed.' + #13#10 + #13#10 +
    'After installation, a second window will open to' + #13#10 +
    'download Python dependencies (~5-8 GB).' + #13#10 +
    'This takes 10-20 minutes on first run.' + #13#10 + #13#10 +
    'Requirements:' + #13#10 +
    '  - Python 3.10 (checked during install)' + #13#10 +
    '  - NVIDIA GPU with 6+ GB VRAM (recommended)' + #13#10 +
    '  - Internet connection for first-time setup' + #13#10 + #13#10 +
    'Click Next to continue.';
end;
