# Interactive Mode Enhancement Plan

## Current State

### CLI Commands Available (23 total)
1. `index` ✓ (available in interactive)
2. `search` ✓ (available in interactive)
3. `detect_fork` ✓ (available in interactive)
4. `fork` ✓ (available in interactive)
5. `status` ✓ (available in interactive)
6. `config_show` ✗ (MISSING)
7. `config_theme` ✗ (MISSING)
8. `reset` ✗ (MISSING)
9. `watch` ✗ (MISSING)
10. `compaction_check` ✗ (MISSING)
11. `compaction_export` ✗ (MISSING)
12. `cluster_analysis` ✗ (MISSING)
13. `tree_build` ✗ (MISSING)
14. `tree_visualize` ✗ (MISSING)
15. `tree_export` ✗ (MISSING)
16. `vault_add` ✗ (MISSING)
17. `vault_list` ✗ (MISSING)
18. `vault_restore` ✗ (MISSING)
19. `vault_search` ✗ (MISSING)
20. `test` ✗ (MISSING)
21. `metrics` ✗ (MISSING)
22. `ab_test_status` ✗ (MISSING)
23. `update_titles` ✗ (MISSING)
24. `interactive` (the command itself)

### Currently in Interactive Shell (10 commands)
1. exit/quit
2. clear
3. status
4. search
5. fork
6. detect_fork
7. index
8. results
9. history
10. help

**Missing: 13 commands need to be added**

---

## Implementation Plan

### Phase 1: Make Interactive Mode Default
**File: `smartfork/src/smartfork/cli.py`**

Modify the main callback to launch interactive mode when no subcommand is provided:

```python
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file path"),
    lite: bool = typer.Option(False, "--lite", "-l", help="Lite mode (minimal resource usage)"),
):
    """SmartFork - AI Session Intelligence for Kilo Code.
    
    Run without commands to start interactive mode.
    """
    if ctx.invoked_subcommand is None:
        # Launch interactive mode
        start_interactive_shell(lite=lite)
        raise typer.Exit()
    
    # Normal CLI mode
    config = get_config()
    if lite:
        config.lite_mode = True
    log_level = "DEBUG" if verbose else config.log_level
    setup_logging(log_level, log_file)
```

### Phase 2: Add Missing Commands to Interactive Shell
**File: `smartfork/src/smartfork/ui/interactive.py`**

Add these do_* methods to SmartForkShell class:

#### Config Commands
```python
def do_config(self, arg: str):
    """Show or set configuration.
    Usage: config [theme <name> | show]
    """
    
def do_theme(self, arg: str):
    """Change color theme.
    Usage: theme <name>
    """
```

#### Database Commands
```python
def do_reset(self, arg: str):
    """Reset the database (WARNING: deletes all data).
    Usage: reset [--force]
    """

def do_watch(self, arg: str):
    """Watch for session changes and index incrementally.
    Usage: watch
    """
```

#### Intelligence Commands
```python
def do_compaction(self, arg: str):
    """Check or export sessions at risk of compaction.
    Usage: compaction [check | export [--dry-run]]
    """

def do_cluster(self, arg: str):
    """Analyze session clusters and find duplicates.
    Usage: cluster
    """

def do_tree(self, arg: str):
    """Build or visualize conversation tree.
    Usage: tree [build | visualize | export]
    """
```

#### Vault Commands
```python
def do_vault(self, arg: str):
    """Manage privacy vault.
    Usage: vault [add <session> | list | restore <session> | search <query>]
    """
```

#### Testing Commands
```python
def do_test(self, arg: str):
    """Run SmartFork tests.
    Usage: test [indexer | search | database | fork]
    """

def do_metrics(self, arg: str):
    """Show success metrics dashboard.
    Usage: metrics [--days N]
    """

def do_abtest(self, arg: str):
    """Show A/B test status.
    Usage: abtest
    """
```

#### Title Commands
```python
def do_titles(self, arg: str):
    """Generate or update session titles.
    Usage: titles [--force] [--dry-run]
    """
```

### Phase 3: Enhance Interactive Shell Features

#### 1. Command Aliases
Add short aliases for common commands:
- `s` → `search`
- `i` → `index`
- `st` → `status`
- `df` → `detect_fork`
- `f` → `fork`
- `t` → `theme`
- `c` → `config`

#### 2. Smart Command Parsing
Handle commands with arguments more naturally:
```
> search python fastapi
> fork abc123
> theme iron
```

#### 3. Auto-completion
Add tab completion for:
- Session IDs (from database)
- File paths
- Theme names
- Command names

#### 4. Command History Persistence
Save command history to `~/.smartfork/history`

---

## Implementation Priority

### High Priority (Core UX)
1. Make interactive mode default (no args)
2. Add `config/theme` commands
3. Add `reset` command
4. Add command aliases (s, i, st, etc.)

### Medium Priority (Advanced Features)
5. Add `compaction` command
6. Add `cluster` command
7. Add `tree` command
8. Add `vault` command

### Lower Priority (Utilities)
9. Add `test` command
10. Add `metrics` command
11. Add `abtest` command
12. Add `titles` command
13. Add `watch` command (with proper handling)

### Nice to Have
14. Auto-completion
15. History persistence
16. Smart command parsing improvements

---

## Testing Checklist

- [ ] Running `smartfork` without args launches interactive mode
- [ ] Running `smartfork <command>` still works as CLI
- [ ] All new commands work in interactive mode
- [ ] Command aliases work
- [ ] Help text is clear for all commands
- [ ] Exit/quit still works properly
- [ ] Theme changes reflect immediately in interactive mode

---

## Files to Modify

1. `smartfork/src/smartfork/cli.py` - Make interactive default
2. `smartfork/src/smartfork/ui/interactive.py` - Add missing commands
3. `smartfork/src/smartfork/ui/__init__.py` - Export updates if needed

---

## Usage Examples (After Implementation)

```bash
# Launch interactive mode (new default)
$ smartfork
SmartFork Interactive Shell
Type 'help' for commands, 'exit' to quit
> 

# Or still use CLI mode
$ smartfork status
$ smartfork search "python fastapi"
$ smartfork index

# In interactive mode
> search python fastapi
> fork abc123
> theme iron
> status
> config theme ember
> cluster
> tree build
> vault list
> metrics
> exit
```
