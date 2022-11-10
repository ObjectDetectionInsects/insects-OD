# Install Git on Windows

### Git for Windows stand-alone installer

1. Download the latest [Git for Windows installer](https://gitforwindows.org/ "Git for Windows installer").
2. When you've successfully started the installer, you should see the Git Setup wizard screen. Follow the Next and Finish prompts to complete the installation. The default options are pretty sensible for most users. 
3. Open a Command Prompt (or Git Bash if during installation you elected not to use Git from the Windows Command Prompt).
4. Run the following commands to configure your Git username and email using the following commands, replacing Emma's name with your own. These details will be associated with any commits that you create:

```bash
$ git config --global user.name "UserName"
$ git config --global user.email "Email"
```
