# Changelog

## [1.3.0](https://github.com/stevearc/gitstack/compare/v1.2.1...v1.3.0) (2025-04-03)


### Features

* transparently update PR target branch, and batch update requests ([feaa75a](https://github.com/stevearc/gitstack/commit/feaa75a5604bd60bbfd49fb8a4c1e419d1a83eb3))


### Bug Fixes

* find previous PRs in other repos via table parsing ([103c703](https://github.com/stevearc/gitstack/commit/103c7036104ad5e8dac0e55e444cd7503e5d8b81))
* stack rebase onto origin/master ([be4ba8c](https://github.com/stevearc/gitstack/commit/be4ba8ce7a768226c1329b267aefb1989a30a499))

## [1.2.1](https://github.com/stevearc/gitstack/compare/v1.2.0...v1.2.1) (2024-01-19)


### Bug Fixes

* the pr command can publish existing PRs ([483afb2](https://github.com/stevearc/gitstack/commit/483afb25c14d4541cb57cabc49f14ba5985f1fe6))

## [1.2.0](https://github.com/stevearc/gitstack/compare/v1.1.0...v1.2.0) (2024-01-19)


### Features

* add support for python 3.8 ([12b8463](https://github.com/stevearc/gitstack/commit/12b8463c8d286cafa604468b16a5e966eb180579))


### Bug Fixes

* allow update command to update or install dev version ([47d3d0e](https://github.com/stevearc/gitstack/commit/47d3d0e2fac288b087a0617197851b6a481d1460))
* pull command will create missing local branches ([9c90da8](https://github.com/stevearc/gitstack/commit/9c90da8a07f6214586a65f5a8aa6cc82a8d71364))
* **pull:** pull all branches in stack ([113b682](https://github.com/stevearc/gitstack/commit/113b6826a38a1054d355b63d4300817b604b8aae))
* stack calculation when master lags origin ([dca5cf5](https://github.com/stevearc/gitstack/commit/dca5cf58db75c9a847b66c795502ee5240cbfa65))

## [1.1.0](https://github.com/stevearc/gitstack/compare/v1.0.0...v1.1.0) (2024-01-18)


### Features

* can link to PRs in a different repository ([845fa9a](https://github.com/stevearc/gitstack/commit/845fa9a492394d7cc25eb73ffd88ef64e3fd96a4))


### Bug Fixes

* add log level choices to help output ([b3a6600](https://github.com/stevearc/gitstack/commit/b3a66001d65bffcce5676ecf9bdbb559af8cdcb5))
* crash when stack has a cycle ([6f6f587](https://github.com/stevearc/gitstack/commit/6f6f5870862b328f37dfdaf3e67a2bf551956a57))
* don't add count to PR title if there is only one entry ([0034370](https://github.com/stevearc/gitstack/commit/003437082d90992dc9c0ee9e148c56985bb22714))
* don't generate PR table if there is only one entry ([c16bddc](https://github.com/stevearc/gitstack/commit/c16bddca37ca66f323c4b52b8a8e472028040c74))

## 1.0.0 (2024-01-02)


### Features

* first working version ([fda00dd](https://github.com/stevearc/gitstack/commit/fda00dd96d7ed6aa867e6db0c664c1058b6cd9ca))
