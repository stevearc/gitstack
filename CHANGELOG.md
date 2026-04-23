# Changelog

## [1.3.0](https://github.com/stevearc/gitstack/compare/v1.2.1...v1.3.0) (2026-04-23)


### Features

* 'rewrite' command to interactive rebase the current branch ([f66bcca](https://github.com/stevearc/gitstack/commit/f66bccabf31506530d4e2510275972a7bed86cd6))
* add --no-push and --force to pr command ([d7b558c](https://github.com/stevearc/gitstack/commit/d7b558c3316979532706f2f8f86f1987e21f2c86))
* clean command ([91a7cdf](https://github.com/stevearc/gitstack/commit/91a7cdf63c3ebc09c5ad0359886e10ffe7c2a210))
* graft command can stack on top of pull request ([baa8162](https://github.com/stevearc/gitstack/commit/baa81626ca6b4a9a38129639b0c1505b914a431e))
* include PR number in list command ([92b85d0](https://github.com/stevearc/gitstack/commit/92b85d0c943c0db0a2381fb8685fb025803364c1))
* new command 'graft' ([6b430d4](https://github.com/stevearc/gitstack/commit/6b430d46321e01faf57c40921d40e34e4dc59410))
* show-base command ([d18ba8f](https://github.com/stevearc/gitstack/commit/d18ba8f9378f609bb682c8706dfba3fa8a6d6b27))
* transparently update PR target branch, and batch update requests ([feaa75a](https://github.com/stevearc/gitstack/commit/feaa75a5604bd60bbfd49fb8a4c1e419d1a83eb3))


### Bug Fixes

* better behavior when deleting merged branches ([d7c4ba0](https://github.com/stevearc/gitstack/commit/d7c4ba0e246285fe544f71e1dd992de225f1bdef))
* dim colors of list commits that are not the branch tip ([3210ab8](https://github.com/stevearc/gitstack/commit/3210ab8d8700b083f27dce28f71c517ee75d7a21))
* don't show ? icon when previous diff in stack is PR-only ([ab27895](https://github.com/stevearc/gitstack/commit/ab27895fcb240e2c89b1a064e86b438beb5f6877))
* find previous PRs in other repos via table parsing ([103c703](https://github.com/stevearc/gitstack/commit/103c7036104ad5e8dac0e55e444cd7503e5d8b81))
* list view uses PR title unless all commits shown ([86b5227](https://github.com/stevearc/gitstack/commit/86b52277aff8d354f624a47b30dce5dc74981f81))
* pr command pushes existing branches after creating PRs ([7e158cb](https://github.com/stevearc/gitstack/commit/7e158cbb07c550fe51ddb6b520c061f13c5e76a1))
* PR table parsing when stack crosses repos ([16cd671](https://github.com/stevearc/gitstack/commit/16cd671e2eef3ba32f16c0eae7a379f63840143b))
* pull request title respects 'publish' setting when created ([a6b026c](https://github.com/stevearc/gitstack/commit/a6b026ccacf6a95fea895789b5f7c8e2c092c956))
* rebasing merged stack onto master is smarter about conflicts ([6213da6](https://github.com/stevearc/gitstack/commit/6213da654a3f691931d373c41739aabe410aa207))
* stack rebase onto origin/master ([be4ba8c](https://github.com/stevearc/gitstack/commit/be4ba8ce7a768226c1329b267aefb1989a30a499))
* use gh pr list instead of gh pr status ([58ca8a5](https://github.com/stevearc/gitstack/commit/58ca8a5d16a064808bf1a7562192417119e83551))

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
