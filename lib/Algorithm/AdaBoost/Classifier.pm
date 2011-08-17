package Algorithm::AdaBoost::Classifier;
use strict;
use warnings;

use Exporter::Lite;

use Class::Accessor::Lite
    rw => [qw(weight division_pos minimum_error)],
    ro => [qw(name)];
use constant EXPLAIN => '';
use constant {
    POSITIVE => 1,
    NEGATIVE => -1,
    NOT_APPLICABLE => 0,
};

our @EXPORT_OK = qw(POSITIVE NEGATIVE NOT_APPLICABLE);


sub new {
    my ($class, $name) = @_;
    return bless {
        name   => $name,
    }, $class;
}

sub classify {
    my ($self, $data) = @_;
    if (defined(my $score = $self->score($data))) {
        return $score < $self->division_pos ? NEGATIVE : POSITIVE;
    } else {
        return NOT_APPLICABLE;
    }
}

sub score {
    my ($self, $data) = @_;
    die 'override this function';
}

sub minimize_error {
    my ($self, $data) = @_;

    my $minimum_error = 1;
    my $best_division_pos = 0;
    my $division_pos = 0;

    while ($division_pos <= 1) {
        for (@$data) {
            my $score = $_->{score}->{$self->name};
            $_->{classified} = defined $score
                ? $score < $division_pos ? NEGATIVE : POSITIVE
                : NOT_APPLICABLE;
        }
        my @misclassified = grep {
            $_->{classified} != NOT_APPLICABLE && $_->{label} != $_->{classified}
        } @$data;
        my $error = 0;
        $error += $_->{weight}->{$self->name} for @misclassified;
        if ($error < $minimum_error) {
            $minimum_error = $error;
            $best_division_pos = $division_pos;
        }

        my $next;
        for (sort {
                $a->{score}->{$self->name} <=> $b->{score}->{$self->name}
            } grep { defined $_->{score}->{$self->name} } @$data) {
            if ($_->{score}->{$self->name} > $division_pos) {
                $division_pos = $_->{score}->{$self->name};
                $next++;
                last;
            }
        }
        $next ? next : last;
    }

    $self->division_pos($best_division_pos);
    return $self->minimum_error($minimum_error);
}


1;
__END__
