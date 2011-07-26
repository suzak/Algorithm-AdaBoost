#!perl
use strict;
use warnings;
use lib qw(lib);
use Algorithm::AdaBoost;


package eg::Classifier::1;
use strict;
use warnings;
use parent 'Algorithm::AdaBoost::Classifier';

sub score {
    my ($self, $data) = @_;
    my $a = (20 - abs($data - 70)) / 20;
    return $a > 0 ? $a : 0;
}

$INC{'eg/Classifier/1.pm'} = __FILE__;

package eg::Classifier::2;
use strict;
use warnings;
use parent 'Algorithm::AdaBoost::Classifier';

sub score {
    my ($self, $data) = @_;
    my $a = (25 - abs($data - 105)) / 25;
    return $a > 0 ? $a : 0;
}

$INC{'eg/Classifier/2.pm'} = __FILE__;

package eg::Classifier::3;
use strict;
use warnings;
use parent 'Algorithm::AdaBoost::Classifier';

sub score {
    my ($self, $data) = @_;
    my $a = (20 - abs($data - 110)) / 20;
    return $a > 0 ? $a : 0;
}

$INC{'eg/Classifier/3.pm'} = __FILE__;

package eg::Classifier::4;
use strict;
use warnings;
use parent 'Algorithm::AdaBoost::Classifier';

sub score {
    my ($self, $data) = @_;
    my $a = (55 - abs($data - 95)) / 55;
    return $a > 0 ? $a : 0;
}

$INC{'eg/Classifier/4.pm'} = __FILE__;


package main;

my $classifier = Algorithm::AdaBoost->new('eg::Classifier' => 1..4);

for (1..200) {
    $classifier->add_data([$_], $_ > 70 && $_ < 130 ? 1 : -1);
}

$classifier->init_train;
$classifier->train;

print $classifier->status;
print "\n";

for (0..9) {
    my $i = int rand 200;
    my $r = $classifier->classify($i);
    print "$i: $r\n";
}
