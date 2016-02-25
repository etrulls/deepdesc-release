
patches = randn( 64, 64, 1, 2 );

save( 'patches.mat', 'patches' );
system( 'th desc.lua' );
desc = load( 'desc.mat' );

desc.x


